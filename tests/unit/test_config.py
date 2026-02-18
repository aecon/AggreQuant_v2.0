"""Unit tests for aggrequant.loaders.config module."""

import tempfile
from pathlib import Path
import pytest
from aggrequant.loaders.config import (
    ChannelConfig,
    SegmentationConfig,
    QualityConfig,
    OutputConfig,
    PipelineConfig,
    create_default_config,
)


class TestChannelConfig:
    """Tests for ChannelConfig dataclass."""

    def test_valid_creation(self):
        """Should create ChannelConfig with valid purpose."""
        ch = ChannelConfig(name="DAPI", pattern="C01", purpose="nuclei")
        assert ch.name == "DAPI"
        assert ch.pattern == "C01"
        assert ch.purpose == "nuclei"

    def test_all_valid_purposes(self):
        """All valid purposes should be accepted."""
        for purpose in ["nuclei", "cells", "aggregates", "other"]:
            ch = ChannelConfig(name="test", pattern="C01", purpose=purpose)
            assert ch.purpose == purpose

    def test_invalid_purpose(self):
        """Invalid purpose should raise ValueError."""
        with pytest.raises(ValueError, match="purpose must be one of"):
            ChannelConfig(name="test", pattern="C01", purpose="invalid")


class TestQualityConfig:
    """Tests for QualityConfig dataclass."""

    def test_default_values(self):
        """Should have correct default values."""
        qc = QualityConfig()
        assert qc.patch_size == (40, 40)
        assert qc.compute_on == []
        assert qc.patch_metrics == ["VarianceLaplacian"]

    def test_tuple_from_list(self):
        """List should be converted to tuple for patch_size."""
        qc = QualityConfig(patch_size=[50, 50])
        assert qc.patch_size == (50, 50)
        assert isinstance(qc.patch_size, tuple)


class TestOutputConfig:
    """Tests for OutputConfig dataclass."""

    def test_default_values(self):
        """Should have correct default values."""
        oc = OutputConfig()
        assert oc.output_subdir == "aggrequant_output"
        assert oc.save_masks is True
        assert oc.overwrite_masks is False

    def test_output_dir_resolved_via_pipeline_config(self):
        """output_dir should be resolved relative to input_dir."""
        config = PipelineConfig(
            input_dir=Path("/data/plate1"),
            plate_format="96",
            output=OutputConfig(output_subdir="results"),
        )
        assert config.output_dir == Path("/data/plate1/results")


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_valid_creation(self):
        """Should create PipelineConfig with valid inputs."""
        config = PipelineConfig(
            input_dir=Path("/data"),
            plate_format="96"
        )
        assert config.input_dir == Path("/data")
        assert config.plate_format == "96"

    def test_string_to_path_conversion(self):
        """String input_dir should be converted to Path."""
        config = PipelineConfig(
            input_dir="/data",
            plate_format="96"
        )
        assert isinstance(config.input_dir, Path)

    def test_valid_plate_formats(self):
        """Both 96 and 384 should be valid plate formats."""
        config_96 = PipelineConfig(input_dir=Path("/data"), plate_format="96")
        config_384 = PipelineConfig(input_dir=Path("/data"), plate_format="384")
        assert config_96.plate_format == "96"
        assert config_384.plate_format == "384"

    def test_invalid_plate_format(self):
        """Invalid plate format should raise ValueError."""
        with pytest.raises(ValueError, match="plate_format must be one of"):
            PipelineConfig(input_dir=Path("/data"), plate_format="48")

    def test_plate_name_default(self):
        """plate_name should default to empty string."""
        config = PipelineConfig(input_dir=Path("/data"), plate_format="96")
        assert config.plate_name == ""

    def test_plate_name_set(self):
        """plate_name should be stored when provided."""
        config = PipelineConfig(
            input_dir=Path("/data"), plate_format="96", plate_name="MyPlate"
        )
        assert config.plate_name == "MyPlate"

    def test_default_nested_configs(self):
        """Should have default nested config objects."""
        config = PipelineConfig(
            input_dir=Path("/data"),
            plate_format="96"
        )
        assert isinstance(config.segmentation, SegmentationConfig)
        assert isinstance(config.quality, QualityConfig)
        assert isinstance(config.output, OutputConfig)


class TestPipelineConfigYamlRoundTrip:
    """Tests for YAML serialization/deserialization."""

    def test_yaml_round_trip(self):
        """Config should survive YAML save/load cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_dir = tmpdir / "data"
            input_dir.mkdir()

            # Create config with various settings
            original = PipelineConfig(
                input_dir=input_dir,
                plate_format="96",
                channels=[
                    ChannelConfig(name="DAPI", pattern="C01", purpose="nuclei"),
                    ChannelConfig(name="GFP", pattern="C02", purpose="aggregates"),
                ],
                control_wells={"negative": ["A01", "A02"]},
                n_workers=8,
                verbose=True,
            )

            # Save and reload
            config_path = tmpdir / "config.yaml"
            original.to_yaml(config_path)
            loaded = PipelineConfig.from_yaml(config_path)

            # Verify key fields match
            assert loaded.plate_name == original.plate_name
            assert loaded.plate_format == original.plate_format
            assert len(loaded.channels) == len(original.channels)
            assert loaded.channels[0].name == "DAPI"
            assert loaded.channels[0].purpose == "nuclei"
            assert loaded.control_wells == original.control_wells
            assert loaded.n_workers == 8
            assert loaded.verbose is True

    def test_overwrite_masks_round_trip(self):
        """overwrite_masks should survive YAML save/load cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_dir = tmpdir / "data"
            input_dir.mkdir()

            original = PipelineConfig(
                input_dir=input_dir,
                plate_format="96",
                output=OutputConfig(overwrite_masks=True),
            )

            config_path = tmpdir / "config.yaml"
            original.to_yaml(config_path)
            loaded = PipelineConfig.from_yaml(config_path)

            assert loaded.output.overwrite_masks is True

    def test_quality_config_tuple_preserved(self):
        """patch_size should be tuple after YAML round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_dir = tmpdir / "data"
            input_dir.mkdir()

            original = PipelineConfig(
                input_dir=input_dir,
                plate_format="96",
            )
            original.quality.patch_size = (50, 50)

            config_path = tmpdir / "config.yaml"
            original.to_yaml(config_path)
            loaded = PipelineConfig.from_yaml(config_path)

            # Should be tuple, not list
            assert loaded.quality.patch_size == (50, 50)
            assert isinstance(loaded.quality.patch_size, tuple)

    def test_none_values_preserved(self):
        """None values should be preserved in YAML round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_dir = tmpdir / "data"
            input_dir.mkdir()

            original = PipelineConfig(
                input_dir=input_dir,
                plate_format="96",
            )
            # Verify aggregate_model_path is None by default
            assert original.segmentation.aggregate_model_path is None

            config_path = tmpdir / "config.yaml"
            original.to_yaml(config_path)
            loaded = PipelineConfig.from_yaml(config_path)

            # Should still be None (not "None" string)
            assert loaded.segmentation.aggregate_model_path is None


class TestCreateDefaultConfig:
    """Tests for create_default_config function."""

    def test_creates_valid_config(self):
        """Should create a valid PipelineConfig."""
        config = create_default_config(
            input_dir=Path("/data"),
            plate_format="96"
        )

        assert isinstance(config, PipelineConfig)
        assert config.input_dir == Path("/data")
        assert config.plate_format == "96"

    def test_default_channels(self):
        """Should have default channels for typical HCS analysis."""
        config = create_default_config(Path("/data"))

        assert len(config.channels) == 3

        # Check channel purposes
        purposes = {ch.purpose for ch in config.channels}
        assert "nuclei" in purposes
        assert "aggregates" in purposes
        assert "cells" in purposes

    def test_384_plate_format(self):
        """Should accept 384-well plate format."""
        config = create_default_config(
            input_dir=Path("/data"),
            plate_format="384"
        )
        assert config.plate_format == "384"
