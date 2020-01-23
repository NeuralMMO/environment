from pdb import set_trace as T

import time
import ray
import sys
import json
import argparse


from functools import partial
from threading import Thread
from random import random

from bokeh.server.server import Server
from bokeh.models import ColumnDataSource
from bokeh.models import Band
from bokeh.models.widgets import RadioButtonGroup
from bokeh.layouts import layout

from bokeh.plotting import curdoc, figure, show
from bokeh.themes import Theme

from tornado import gen
from tornado.ioloop import IOLoop

from config import *
from signal import signal, SIGTERM, SIGINT
from time import gmtime, strftime

# Parse arguments at start of script
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--log', action="store_true",
                    help='Log data on visualizer exit. Default file is timestamp, filename overwrite with --name')
parser.add_argument('--name',
                    help='Name of file to save json data to')
parser.add_argument('--load', action="store_true",
                    help='Loads saved json into visualizer with name specified by --name')
parser.add_argument("--xaxis", default='tick', help='Name of key in data to be used for the x axis of the graph, defaults to \'tick\'')

arguments = parser.parse_args()
LOAD_DATA = {}
LOG = True if arguments.log else False
LOAD = True if arguments.load else False
SCALES = []
XAXIS = ''

if arguments.load:
    if not arguments.name:
        print("Please specify file name to load with --name")
        sys.exit(0)
    with open(arguments.name, 'r') as myfile:
        LOAD_DATA = myfile.read()
    LOAD_DATA = json.loads(LOAD_DATA)
    SCALES = LOAD_DATA.pop('SCALES')
    ITEMS = LOAD_DATA.pop('ITEMS')
    XAXIS = LOAD_DATA.pop('XAXIS')



class MarketVisualizer:
    def __init__(self,
                 data: dict = {},
                 keys: list = [],
                 scales: list = [1],
                 history_len: int = 256,
                 title: str = "NeuralMMO Market Data",
                 x: str = "tick",
                 ylabel: str = "Dummy Values"):
        """Visualizes a stream of data with threaded refreshing. To
        add items, initialize using 'keys' kwarg or add to packet in
        stream()

        Args:
            keys        : List of object names (str) to be displayed on market
            history_len : How far back to plot data
            scales      : Scales for visualization
            title       : Title of graph
            x           : Name of x axis data
            ylabel      : Name of y axis on plot
        """
        self.colors = 'blue red green yellow orange purple brown white'.split()
        self.data        = data

        # Market stores data in a dictionary of lists
        # MarketVisualizer stores the data for each view as a dictionary,
        # inside another dictionary with keys of each scale.

        self.scales = scales


        self.history_len = history_len
        self.title       = title
        self.keys        = keys

        self.ylabel      = ylabel
        self.x           = x

        # TODO figure out a better way to ensure unique colors
        assert len(self.keys) <= len(self.colors), 'Limited color pool'
        # data initialization
        if self.data == {}:
            for scale in self.scales:
                self.data[scale] = {}
            for scale in self.scales:
                for i, key in enumerate(self.keys):
                    self.data[scale][key] = []
                    self.data[scale][key+'lower'] = []
                    self.data[scale][key+'upper'] = []
                self.data[scale][self.x] = []
                # Isn't necessary, but makes code for packet handling nicer
                self.data[scale][self.x + 'upper'] = []
                self.data[scale][self.x + 'lower'] = []
        if type(self.scales[0]) != str:
            self.scales = [str(s) for s in self.scales]
        self.scale = self.scales[0]
        print("scale", self.scale)
        print(self.data)
        print(self.data[self.scale])

    def init(self, doc):
        # Source must only be modified through a stream
        self.source = ColumnDataSource(data=self.data[self.scale])

        # Enable theming
        theme = Theme('theme.yaml')
        doc.theme = theme
        self.doc = doc


        fig = figure(
           plot_width=600,
           plot_height=400,
           tools='xpan,xwheel_zoom, xbox_zoom, reset',
           title='Neural MMO: Market Data',
           x_axis_label=self.x,
           y_axis_label=self.ylabel)

        # Initialize plots
        for i, key in enumerate(self.keys):
            fig.line(
                source=self.source,
                x=self.x,
                y=key,
                color=self.colors[i],
                line_width=LINE_WIDTH,
                legend_label=key)

            band = Band(
               source=self.source,
               base=self.x,
               lower=key+'lower',
               upper=key+'upper',
               level='underlay',
               line_color=self.colors[i],
               line_width=1,
               line_alpha=0.2,
               fill_color=self.colors[i],
               fill_alpha=0.2)
            fig.add_layout(band) 

        def switch_scale(attr, old, new):
            """Callback for RadioButtonGroup to switch tick scale
            and refresh document
            Args:
                attr: variable to be changed, in this case 'active'
                old: old index of active button
                new: new index of active button
            """
            self.scale = self.scales[new]
            self.source.data = self.data[self.scale]

        self.timescales = RadioButtonGroup(labels=[str(scale) for scale in self.scales],
                                           active=self.scales.index(self.scale))
        self.timescales.on_change('active', switch_scale)

        self.structure = layout([[self.timescales], [fig]])
        self.doc.add_root(self.structure)

        self.fig = fig

    def stream(self, packet: dict):
        '''Wrapper function for source.stream to enable
        adding new items mid-stream. Overwrite graph
        with new figure if packet has different keys.

        Args:
            packet: dictionary of singleton lists'''

        # Stream items & append data if no new entries
        if packet.keys() == self.data[self.scale].keys():
            for scale in self.scales:
                # Skip over irrelevant scales
                if packet[self.x][0] % scale != 0:
                    continue
                for key, val in packet.items():
                    self.data[scale][key].append(val[0])
            # Once data is added, stream if tick is member
            # of current scale
            if packet[self.x][0] % self.scale == 0:
                self.source.stream(packet, self.history_len)
            return

        # Add new data entry & refresh document if new entry
        for scale in self.scales:
            for key, val in packet.items():

                # Pads new data with 0 entry before adding
                if key not in self.data[scale].keys():
                    pad_len = len(self.data[scale][self.x])
                    self.data[scale][key] = ([0] * pad_len)

                    # Adds new entry value if relevant
                    if packet[self.x][0] % scale == 0 and pad_len > 0:
                        self.data[scale][key][-1] = val[0]
                    # Adds new entry to keys if valid key
                    if key[-5:] not in ['lower', 'upper'] and key not in self.keys + [self.x]:
                        self.keys.append(key)

                # If not new entry, add to data
                elif packet[self.x][0] % scale != 0:
                    self.data[scale][key].append(val[0])

        # Refreshes document to add new item
        self.doc.remove_root(self.structure)
        self.init(self.doc)


@ray.remote
class BokehServer:
    def __init__(self, market, *args, **kwargs):
        """ Runs an asynchronous Bokeh data streaming server.
      
        Args:
           market : The market to visualize
           args   : Additional arguments
           kwargs : Additional keyword arguments
        """
        self.visu   = MarketVisualizer(*args, **kwargs)
        self.market = market

        server = Server(
                {'/': self.init},
                io_loop=IOLoop.current(),
                port=PORT,
                num_procs=1)

        self.server = server
        self.thread = None
        self.tick   = 0
        self.packet = {}

        if 'scales' in kwargs and type(kwargs['scales']) is list:
            self.scales = kwargs['scales']
        else:
            self.scales = [1]

        server.start()

        server.io_loop.add_callback(server.show, "/")
        server.io_loop.start()

    def init(self, doc):
        '''Initialize document and threaded update loop

        Args:
           doc: A Bokeh document
        '''
        self.visu.init(doc)
        self.doc = doc

        self.thread = Thread(target=self.update, args=[])
        self.thread.start()
        self.started = True

    def update(self):
        '''Blocking update call to be run in a separate thread

        Ingests packets from a remote market and streams to Bokeh client'''
        self.n = 0
        while True:
            # Wait for thread to initialize
            time.sleep(0.05)
            if self.thread is None:
               continue 

            # Get remote market data

            if ray.get(self.market.getShutdown.remote()):
                self.market.setData.remote(self.visu.data)
                sys.exit(0)

            packet = ray.get(self.market.getData.remote())
            if packet is None:
               continue
            self.packet = packet.copy() if type(packet) == dict else {}

            # Ingest market data, add upper and lower values for each scale
            # self.packet streams latest tick to visualizer
            for key, val in packet.items():
                # Add to packet
                self.packet[key] = [self.packet[key]]
                self.packet[key + 'lower'] = [val - 0.1]
                self.packet[key + 'upper'] = [val + 0.1]

            # Stream to Bokeh client
            self.doc.add_next_tick_callback(partial(self.stream))
            self.tick += 1

    @gen.coroutine
    def stream(self):
        '''Stream current data buffer to Bokeh client'''
        # visu is visualizer object
        self.visu.stream(self.packet)



@ray.remote
class Middleman:
    def __init__(self):
        '''Remote data buffer for two processes to dump and recv data.
        Interacts with Market and BokehServer.

        This is probably not safe'''
        self.data = None
        self.shutdown = 0
        
    def getData(self):
        '''Get data from buffer

        Returns:
           data: From buffer
        '''
        data = self.data
        self.data = None
        return data

    def setData(self, data):
        '''Set buffer data

        Args:
           data: To set buffer
        '''
        self.data = data.copy()

    def setShutdown(self):
        self.shutdown = 1

    def getShutdown(self):
        return self.shutdown

class Market:
    def __init__(self, items, middleman):
        '''Dummy market emulator

        Args: 
           items     : List of item keys
           middleman : A Middleman object'''
        self.middleman = middleman
        self.items     = items
        
        self.keys = items
        # Data contains market history for lowest values
        self.data = {}
        self.tick = 0

        self.data['tick'] = 0
        for i, key in enumerate(self.keys):
            self.data[key] = 0

    def update(self):
        '''Updates market data and propagates to Bokeh server
        # update data and send to middle man
        Note: best to update all at once. Current version may cause bugs'''
        for key, val in self.data.items():
            if key == 'tick':
                self.data[key] = self.tick
            else:
                self.data[key] = val + 0.2*(random() - 0.5)

        self.tick += 1
        
        # dummy code to inject new items, it's as easy as just
        # adding new data to the market

        if self.tick % 100 == 0:
            self.data['new_item'] = 5
        if self.tick % 130 == 0:
            self.data['new_item_2'] = 5
        # update if not shutting down visualizer
        if not ray.get(middleman.getShutdown.remote()):
            self.middleman.setData.remote(self.data)

# Example setup
PORT=5009
ray.init()

ITEMS = ['Food', 'Water', 'Health', 'Melee', 'Range', 'Mage']
# ITEMS = []    <-- Also a valid setup, new items can be added dynamically
SCALES = [1, 10, 100, 1000]


def log_on_exit(*args):
    '''Logging function called when visualizer is terminated '''
    # Prevents multiple exit signals from printing multiple logs
    middleman.setShutdown.remote()
    time.sleep(2)
    signal(SIGTERM, clean_exit)
    signal(SIGINT, clean_exit)
    
    if arguments.name:
        filename = arguments.name
    else:
        filename = strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '.json'
    
    logdata = ray.get(middleman.getData.remote())
    logdata['XAXIS'] = arguments.xaxis
    logdata['SCALES'] = SCALES
    logdata['ITEMS'] = ITEMS
    f = open(filename, "w")
    f.write(json.dumps(logdata, indent=4))
    f.close()
    sys.exit(0)

def clean_exit(*args):
    sys.exit(0)


if LOG:
    signal(SIGTERM, log_on_exit)
    signal(SIGINT, log_on_exit)


middleman  = Middleman.remote()
market     = Market(ITEMS, middleman)
visualizer = BokehServer.remote(middleman, data=LOAD_DATA, keys=ITEMS, scales=SCALES)

if LOAD:
    input()
    sys.exit(0)


while True:
  time.sleep(0.1)
  market.update()
