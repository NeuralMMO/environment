#pylint: disable=invalid-name, unused-argument
from nmmo.task.predicate.core import predicate
from nmmo.task.group import Group
from nmmo.task.game_state import GameState

@predicate
def HoardGold(gs: GameState,
              subject: Group,
              amount: int):
  """True iff the summed gold of all teammate is greater than or equal to amount.
  """
  return subject.gold.sum() / amount

#######################################
# Event-log based predicates
#######################################

@predicate
def EarnGold(gs: GameState,
             subject: Group,
             amount: int):
  """ True if the total amount of gold earned is greater than or equal to amount.
  """
  return subject.event.EARN_GOLD.gold.sum() / amount

@predicate
def SpendGold(gs: GameState,
              subject: Group,
              amount: int):
  """ True if the total amount of gold spent is greater than or equal to amount.
  """
  return subject.event.BUY_ITEM.gold.sum() / amount

@predicate
def MakeProfit(gs: GameState,
               subject: Group,
               amount: int):
  """ True if the total amount of gold earned-spent is greater than or equal to amount.
  """
  profits = subject.event.EARN_GOLD.gold.sum()
  costs = subject.event.BUY_ITEM.gold.sum()
  return  (profits-costs) / amount
