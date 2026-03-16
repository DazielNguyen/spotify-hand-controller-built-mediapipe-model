"""Dataset module placeholder."""


class LandmarkSample:
    """Placeholder sample container for future dataset work."""


def load_annotation_samples(dataset_root, annotations_path):
    """Placeholder for annotation loading."""
    raise NotImplementedError("TODO: implement dataset annotation loading")


def load_image(path, image_size):
    """Placeholder for image loading and preprocessing."""
    raise NotImplementedError("TODO: implement image loading")


def prepare_training_arrays(dataset_root, annotations_path, image_size=(128, 128)):
    """Placeholder for training array preparation."""
    raise NotImplementedError("TODO: implement training array preparation")


def summarize_annotations(dataset_root, annotations_path):
    """Placeholder for annotation summary logic."""
    raise NotImplementedError("TODO: implement annotation summary")