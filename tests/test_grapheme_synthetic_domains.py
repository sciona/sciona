import numpy as np

from scripts.grapheme_synthetic_domains import complete_populations, KNOWN, UNSEEN


def test_unseen_font_class_is_absent_from_handwriting_training():
    data = complete_populations()
    assert set(data['hand'].labels) == set(KNOWN)
    assert UNSEEN not in data['hand'].labels
    for population in (*data['fonts'], data['calibration'], data['query']):
        assert set(population.labels) == {*KNOWN, UNSEEN}
    assert np.flatnonzero(data['known_class_mask']).tolist() == list(KNOWN)
    assert not data['known_class_mask'][UNSEEN]
    np.testing.assert_array_equal(np.asarray(KNOWN)[data['ood_training_labels']], data['hand'].labels)


def test_populations_are_reproducible_distinct_and_read_only():
    first, second = complete_populations(), complete_populations()
    populations = [first['hand'], *first['fonts'], first['calibration'], first['query']]
    repeats = [second['hand'], *second['fonts'], second['calibration'], second['query']]
    seen = set()
    for population, repeat in zip(populations, repeats):
        np.testing.assert_array_equal(population.images, repeat.images)
        assert not population.images.flags.writeable and not population.labels.flags.writeable
        assert population.images.dtype == np.uint8 and population.images.shape[1:] == (137, 236)
        assert 0 <= population.images.min() < population.images.max() <= 255
        for image in population.images:
            # Exclusively constructed pixels: no real dataset content is read.
            pixels = image.tobytes()
            assert pixels not in seen
            seen.add(pixels)
