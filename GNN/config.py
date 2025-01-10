'''
Configuration options for RegGNN pipeline.
'''


class Config:
    # SYSTEM OPTIONS
    DATA_FOLDER = './simulated_data/'  # path to the folder data will be written to and read from
    RESULT_FOLDER = './'  # path to the folder data will be written to and read from

    # EVALUATION OPTIONS
    K_FOLDS = 3  # number of cross validation folds

    # REGGNN OPTIONS
    class RegGNN:
        NUM_EPOCH = 150  # number of epochs the process will be run for
        LR = 3e-5  # learning rate
        WD = 5e-4  # weight decay
        DROPOUT = 0.1  # dropout rate


    class SampleSelection:
        #changed from True to False
        SAMPLE_SELECTION = False  # whether or not to apply sample selection
        K_LIST = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # list of k values for sample selection
        N_SELECT_SPLITS = 10  # number of folds for the nested sample selection cross validation

    # RANDOMIZATION OPTIONS
    DATA_SEED = 1  # random seed for data creation
    MODEL_SEED = 1  # random seed for models
    SHUFFLE = True  # whether to shuffle or not
