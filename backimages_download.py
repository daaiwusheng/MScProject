import fiftyone as fo
import fiftyone.zoo as foz


def get_validation():
    # List available zoo datasets
    print(foz.list_zoo_datasets())

    #
    # Load the COCO-2017 validation split into a FiftyOne dataset
    #
    # This will download the dataset from the web, if necessary
    #
    dataset = foz.load_zoo_dataset("coco-2017", split="validation")

    # Give the dataset a new name, and make it persistent so that you can
    # work with it in future sessions
    dataset.name = "coco-2017-validation-example"
    dataset.persistent = True

    # Visualize the in the App
    session = fo.launch_app(dataset)


def get_test():
    dataset = foz.load_zoo_dataset("coco-2017", split="test")

    # Give the dataset a new name, and make it persistent so that you can
    # work with it in future sessions
    dataset.name = "coco-2017-test"
    dataset.persistent = True

    # Visualize the in the App

    session = fo.launch_app(dataset)


if __name__ == '__main__':
    get_test()
