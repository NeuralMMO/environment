from pdb import set_trace as T

import time
import ray

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

# market plans
# click on objects in market to display stats about them
# market overview tab
# -> trade bandwidth, demand, gdp

class MarketVisualizer:
    def __init__(self,
                 keys,
                 scales: list = [1],
                 history_len: int = 256,
                 title: str = "NeuralMMO Market Data",
                 x: str = "tick",
                 ylabel: str = "Dummy Values"):
        """Visualizes a stream of data with threaded refreshing

        Args:
            keys        : List of object names (str) to be displayed on market
            history_len : How far back to plot data
            scales      : Scales for visualization
            title       : Title of graph
            x           : Name of x axis data
            ylabel      : Name of y axis on plot
        """
        self.colors = 'blue red green yellow orange purple'.split()
        self.data        = {}

        # Market stores data in a dictionary of lists
        # MarketVisualizer stores the data for each view as a dictionary,
        # inside another dictionary with keys of each scale.

        self.scales = scales
        for scale in self.scales:
            self.data[scale] = {}
        self.scale = self.scales[0]


        self.history_len = history_len
        self.title       = title
        self.keys        = keys

        self.ylabel      = ylabel
        self.x           = x

        # TODO figure out a better way to ensure unique colors
        assert len(self.keys) <= len(self.colors), 'Limited color pool'

        # data initialization
        for scale in self.scales:
            for i, key in enumerate(self.keys):
                self.data[scale][key] = []
                self.data[scale][key+'lower'] = []
                self.data[scale][key+'upper'] = []
            self.data[scale][self.x] = []
            # Isn't necessary, but makes code for packet handling nicer
            self.data[scale][self.x + 'upper'] = []
            self.data[scale][self.x + 'lower'] = []


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
            """
            Callback for RadioButtonGroup to switch tick scale
            and refresh document

            Args:
                attr: variable to be changed, in this case 'active'
                old: old index of active button
                new: new index of active button
            """

            self.scale = self.scales[new]
            # redraw graph
            self.doc.remove_root(structure)
            self.init(self.doc)

        self.timescales = RadioButtonGroup(
            labels=[str(scale) for scale in self.scales],
            active=self.scales.index(self.scale),
            background=RADIO_BACKGROUND)
        self.timescales.on_change('active', switch_scale)

        structure = layout([[self.timescales], [fig]])
        self.doc.add_root(structure)

        self.fig = fig

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
            packet = ray.get(self.market.getData.remote())
            if packet is None:
               continue
            self.packet = packet.copy() if type(packet) == dict else {}

            # Ingest market data, add upper and lower values for each scale
            # self.visu.data stores all data
            # self.packet streams latest tick to visualizer

            for key, val in packet.items():
                for scale in self.scales:
                    # Ignore any irrelevant buffers
                    if packet['tick'] % scale != 0:
                        continue
                    self.visu.data[scale][key].append(val)
                    self.visu.data[scale][key + 'lower'].append(val - 0.1)
                    self.visu.data[scale][key + 'upper'].append(val + 0.1)
                self.packet[key] = [self.packet[key]]
                self.packet[key + 'lower'] = [val - 0.1]
                self.packet[key + 'upper'] = [val + 0.1]

            # Stream to Bokeh client
            self.doc.add_next_tick_callback(partial(self.stream))
            self.tick += 1

    @gen.coroutine
    def stream(self):
        '''Stream current data buffer to Bokeh client'''
        # TODO update with better buffers
        # visu is visualizer object
        if self.tick % self.visu.scale == 0:
            self.visu.source.stream(self.packet, self.visu.history_len)


@ray.remote
class Middleman:
    def __init__(self):
        '''Remote data buffer for two processes to dump and recv data.
        Interacts with Market and BokehServer.

        This is probably not safe'''
        self.data = None
        
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

class Market:
    def __init__(self, items, middleman):
        '''Dummy market emulator

        Args: 
           items     : List of item keys
           middleman : A Middleman object'''
        self.middleman = middleman
        self.items     = items

        self.keys = items
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
            self.data[key] = val + 0.2*(random() - 0.5)
            if key == 'tick':
                self.data[key] = self.tick

        self.tick += 1
        self.middleman.setData.remote(self.data)

# Example setup
PORT=5009
ray.init()
ITEMS = ['Food', 'Water', 'Health', 'Melee', 'Range', 'Mage']
SCALES = [1, 10, 100, 1000]

middleman  = Middleman.remote()
market     = Market(ITEMS, middleman)
visualizer = BokehServer.remote(middleman, ITEMS, scales=SCALES)

while True:
  time.sleep(0.1)
  market.update()
