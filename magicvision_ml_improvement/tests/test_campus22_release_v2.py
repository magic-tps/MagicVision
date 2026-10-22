from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import campus22_schema as schema
from campus22_release_v2 import assign_groups, classify_missing, exact_duplicate_evidence
from data_pipeline import record


def make_row(tmp_path, name, boxes, split=None, group=None, color='red', **extra):
    image = tmp_path / f'{name}.png'; Image.new('RGB', (40, 40), color).save(image)
    label = tmp_path / f'{name}.txt'; schema.write_labels(label, boxes)
    return record(image, label, 'fixture', group=group or name, original_split=split,
                  taxonomy=schema.TAXONOMY, **extra)


def test_group_assignment_keeps_related_rows_together_and_prefers_test(tmp_path):
    rows = [make_row(tmp_path, 'a', [[5, .5, .5, .2, .2]], 'train', 'session', 'red'),
            make_row(tmp_path, 'b', [[5, .5, .5, .2, .2]], 'test', 'session', 'blue')]
    plans = assign_groups(rows)
    assert len(plans) == 1 and plans[0]['conflict_before']
    assert plans[0]['resolved_split'] == 'test' and not plans[0]['leakage_after']
    assert {row['resolved_split'] for row in rows} == {'test'}
    assert len({row['group_id'] for row in rows}) == 1


def test_empty_source_classification_is_not_promoted_to_negative(tmp_path):
    row = make_row(tmp_path, 'empty', [], candidate_classes=['cone'], negative_candidate=True)
    result = classify_missing(row)
    assert result['classification'] == 'C_POSSIBLE_OBJECT_WITHOUT_ANNOTATION'
    assert result['status'] == 'NEEDS_HUMAN_REVIEW'
    assert result['automatic_or_manual'] == 'manual'


def test_existing_verified_negative_is_the_only_automatic_negative(tmp_path):
    row = make_row(tmp_path, 'negative', [], verified_negative=True)
    result = classify_missing(row)
    assert result['classification'] == 'A_VERIFIED_NEGATIVE'
    assert result['status'] == 'RESOLVED'


def test_exact_duplicate_with_different_annotations_requires_reconciliation(tmp_path):
    first = make_row(tmp_path, 'same', [[13, .5, .5, .4, .8]], color='red')
    other_label = tmp_path / 'other.txt'; schema.write_labels(other_label, [[14, .5, .5, .4, .8]])
    second = {**first, 'id': 'other', 'label': str(other_label)}
    evidence = exact_duplicate_evidence([first, second])
    assert len(evidence) == 1 and not evidence[0]['same_annotation']
    assert evidence[0]['action'] == 'NEEDS_HUMAN_RECONCILIATION'


def test_group_ids_are_stable_across_input_order(tmp_path):
    rows = [make_row(tmp_path, 'one', [[0, .5, .5, .2, .2]], group='g', color='red'),
            make_row(tmp_path, 'two', [[0, .5, .5, .2, .2]], group='g', color='blue')]
    assign_groups(rows); expected = {row['id']: row['group_id'] for row in rows}
    reversed_rows = [dict(row) for row in reversed(rows)]
    assign_groups(reversed_rows)
    assert {row['id']: row['group_id'] for row in reversed_rows} == expected
