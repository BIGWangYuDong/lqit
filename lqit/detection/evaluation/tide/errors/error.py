# Modified from https://github.com/dbolya/tide
# This work is licensed under MIT license.
from typing import Union


class Error:
    """A base class for all error types."""

    def fix(self) -> Union[tuple, None]:
        """Returns a fixed version of the AP data point for this error or None
        if this error should be suppressed.

        Returns:
            tuple: (score:float, is_positive:bool, info:dict)
        """
        raise NotImplementedError

    def unfix(self) -> Union[tuple, None]:
        """Returns the original version of this data point."""

        if hasattr(self, 'pred'):
            # If an ignored instance is an error, it's not in the data
            # point list, so there's no "unfixed" entry
            if self.pred['used'] is None:
                return None
            else:
                return (self.pred['class'], (self.pred['score'], False,
                                             self.pred['info']))
        else:
            return None

    def get_id(self) -> int:
        """Get index."""
        if hasattr(self, 'pred'):
            return self.pred['_id']
        elif hasattr(self, 'gt'):
            return self.gt['_id']
        else:
            return -1


class BestGTMatch:
    """Some errors are fixed by changing false positives to true positives. The
    issue with fixing these errors naively is that you might have multiple
    errors attempting to fix the same GT. In that case, we need to select which
    error actually gets fixed, and which others just get suppressed (since we
    can only fix one error per GT).

    To address this, this class finds the prediction with the hiighest score
    and then uses that as the error to fix, while suppressing all other errors
    caused by the same GT.
    """

    def __init__(self, pred, gt) -> None:
        self.pred = pred
        self.gt = gt

        if self.gt['used']:
            self.suppress = True
        else:
            self.suppress = False
            self.gt['usable'] = True

            score = self.pred['score']

            if 'best_score' not in self.gt:
                self.gt['best_score'] = -1

            if self.gt['best_score'] < score:
                self.gt['best_score'] = score
                self.gt['best_id'] = self.pred['_id']

    def fix(self) -> Union[tuple, None]:
        if self.suppress or self.gt['best_id'] != self.pred['_id']:
            return None
        else:
            return self.pred['score'], True, self.pred['info']
