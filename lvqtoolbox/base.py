# TODO: LVQClassifier - base class for LVQClassifier
# Abstract classes are not Pythonian apparently. However, common functionality which we always need can still be put in
# a common base class.


# TODO: Think about this... keep it in a common class or common module/package
class LVQClassifier(object):

    def other_common_function(self):
        pass

    def _conditional_mean(self):
        pass

    # TODO: Change this to something more configurable. For now Conditional mean
    def init_prototypes(self, prototype_labels, data, data_labels):
        pass
