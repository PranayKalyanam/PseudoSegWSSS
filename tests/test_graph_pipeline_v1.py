from __future__ import annotations

from configs.config import get_config

from loaders.patient_loader import PatientLoader
from loaders.image_loader import ImageLoader

from methods.graph.patch_group_serializer import PatchGroupSerializer
from methods.patch.coordinate_generator import CoordinateGenerator
from methods.patch.patch_extractor import PatchExtractor
from methods.patch.tissue_patch_filter import TissuePatchFilter
from methods.patch.weak_label_generator import WeakLabelGenerator
from methods.patch.patch_metadata_generator import PatchMetadataGenerator
from methods.patch.patch_collection_builder import PatchCollectionBuilder

from methods.visualization.patch_visualizer import PatchVisualizer



from methods.graph.graph_builder import GraphBuilder
from methods.graph.graph_visualizer import GraphVisualizer
from methods.graph.patch_group_builder import PatchGroupBuilder

from utils.logger import get_logger


def main():

    logger = get_logger("GraphPipelineTest", log_dir="logs")

    logger.info("=" * 80)
    logger.info("GRAPH V1 TEST")
    logger.info("=" * 80)

    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------

    config = get_config()

    # --------------------------------------------------
    # Patient Discovery
    # --------------------------------------------------

    patient_loader = PatientLoader(
        config=config,
        logger=logger,
    )

    patients = patient_loader.load_all()

    logger.info("Patients discovered : %d", len(patients))

    # --------------------------------------------------
    # Image Loading
    # --------------------------------------------------

    image_loader = ImageLoader(
        config=config,
        logger=logger,
    )

    patients = image_loader.load_all(
        patients,
    )

    # ==================================================
    # Process Every Patient
    # ==================================================

    for patient in patients:

        logger.info("")
        logger.info("=" * 80)
        logger.info("PATIENT : %s", patient.patient_id)
        logger.info("=" * 80)

        logger.info(
            "Original Image Shape : %s",
            patient.image.shape,
        )

        logger.info(
            "Original Mask Shape  : %s",
            patient.mask.shape,
        )

        ##################################################
        # Coordinate Generator
        ##################################################

        logger.info("")
        logger.info("-" * 80)
        logger.info("Coordinate Generator")
        logger.info("-" * 80)

        coordinate_generator = CoordinateGenerator(
            config=config,
            patch_size=config.patch_size,
            overlap=config.overlap,
        )

        coordinates = coordinate_generator.generate(
            image=patient.image,
            tissue_regions=[
                type(
                    "DummyRegion",
                    (),
                    {
                        "bounding_box": type(
                            "BBox",
                            (),
                            {
                                "x": 0,
                                "y": 0,
                                "width": patient.image.shape[1],
                                "height": patient.image.shape[0],
                            },
                        )()
                    },
                )()
            ],
        )

        logger.info(
            "Coordinates Generated : %d",
            len(coordinates),
        )

        for i, coordinate in enumerate(coordinates[:10]):

            logger.info(
                "[%04d] (%d,%d,%d,%d)",
                i,
                coordinate.x,
                coordinate.y,
                coordinate.width,
                coordinate.height,
            )

        ##################################################
        # Patch Extraction
        ##################################################

        logger.info("")
        logger.info("-" * 80)
        logger.info("Patch Extraction")
        logger.info("-" * 80)

        extractor = PatchExtractor()

        patches = extractor.extract(
            coordinates=coordinates,
            image=patient.image,
            annotation=patient.mask,
            source_filename=patient.image_filename,
        )

        logger.info(
            "Extracted Patches : %d",
            len(patches),
        )

        ##################################################
        # Tissue Filter
        ##################################################

        logger.info("")
        logger.info("-" * 80)
        logger.info("Tissue Filter")
        logger.info("-" * 80)

        tissue_filter = TissuePatchFilter(
            threshold=0.50,
        )

        patches = tissue_filter.filter(
            patches,
        )

        logger.info(
            "Remaining Patches : %d",
            len(patches),
        )

        ##################################################
        # Weak Labels
        ##################################################

        logger.info("")
        logger.info("-" * 80)
        logger.info("Weak Label Generation")
        logger.info("-" * 80)

        weak_label_generator = WeakLabelGenerator()

        patches = weak_label_generator.generate(
            patches,
        )

        ##################################################
        # Metadata
        ##################################################

        logger.info("")
        logger.info("-" * 80)
        logger.info("Metadata Generation")
        logger.info("-" * 80)

        metadata_generator = PatchMetadataGenerator()

        patches = metadata_generator.generate(
            patches,
            original_filename=patient.image_filename,
        )

        ##################################################
        # Dataset Builder
        ##################################################

        logger.info("")
        logger.info("-" * 80)
        logger.info("Patch Collection Builder")
        logger.info("-" * 80)

        builder = PatchCollectionBuilder()

        dataset = builder.build(
            patient=patient,
            patches=patches,
        )

        logger.info(
            "Total Patches : %d",
            dataset.number_of_patches,
        )

        logger.info(
            "Average Tissue : %.4f",
            dataset.statistics.average_tissue_percentage,
        )

        logger.info(
            "Average Patch Area : %.2f",
            dataset.statistics.average_patch_area,
        )

        logger.info(
            "Class Distribution : %s",
            dataset.statistics.class_distribution,
        )

        ##################################################
        # Print Patch Information
        ##################################################

        logger.info("")
        logger.info("-" * 80)
        logger.info("Patch Details")
        logger.info("-" * 80)

        # Create an output directory for the raw patches
        import os
        import cv2  # or torch, numpy, etc.
        patch_img_dir = f"outputs/train/{patient.patient_id}/imgs"
        os.makedirs(patch_img_dir, exist_ok=True)
        patch_mask_dir = f"outputs/train/{patient.patient_id}/mask"
        os.makedirs(patch_mask_dir, exist_ok=True)

        for patch in dataset.patches:
            # Save the image patch
            # img_path = os.path.join(patch_img_dir, f"patch_{patch.patch_id}_img.png")
            img_path = os.path.join(patch_img_dir, patch.metadata.image_filename)

            cv2.imwrite(img_path, patch.image_patch)
            
            # Save the mask patch
            # mask_path = os.path.join(patch_mask_dir, f"patch_{patch.patch_id}_mask.png")
            mask_path = os.path.join(patch_mask_dir, patch.metadata.annotation_filename)

            cv2.imwrite(mask_path, patch.annotation_patch)

            logger.info(
                "Patch %04d",
                patch.patch_id,
            )

            logger.info(
                "Coordinate : (%d,%d,%d,%d)",
                patch.coordinate.x,
                patch.coordinate.y,
                patch.coordinate.width,
                patch.coordinate.height,
            )

            logger.info(
                "Image Shape : %s",
                patch.image_patch.shape,
            )

            logger.info(
                "Mask Shape  : %s",
                patch.annotation_patch.shape,
            )

            logger.info(
                "Tissue %% : %.2f",
                patch.tissue_percentage * 100,
            )

            logger.info(
                "Detected Classes : %s",
                patch.detected_classes,
            )

            logger.info("")



        logger.info("")
        logger.info("=" * 80)
        logger.info("GRAPH CONSTRUCTION")
        logger.info("=" * 80)
        
        ##################################################
        # Graph Builder
        ##################################################

        graph_builder = GraphBuilder(
            config=config,
            logger=logger,
        )

        graph = graph_builder.build(
            dataset,
        )


        ##################################################
        # Graph Serializer
        ##################################################

        logger.info("")
        logger.info("=" * 80)
        logger.info("GRAPH SERIALIZATION")
        logger.info("=" * 80)

        from methods.graph.graph_serializer import GraphSerializer

        graph_serializer = GraphSerializer(
            logger=logger,
        )

        graph_json_path = graph_serializer.save(
            graph=graph,
            output_directory=f"outputs/train/{patient.patient_id}",
        )

        logger.info(
            "Graph JSON : %s",
            graph_json_path,
        )

        ##################################################
        # Graph Statistics
        ##################################################

        # statistics = GraphStatistics.compute(
        #     graph,
        # )

        logger.info("Nodes              : %d", graph.statistics.number_of_nodes)
        logger.info("Edges              : %d", graph.statistics.number_of_edges)
        logger.info("Average Degree     : %.2f", graph.statistics.average_degree)
        logger.info("Connected Components : %d", graph.statistics.connected_components)
        logger.info("Isolated Nodes     : %d", graph.statistics.isolated_nodes)
        # logger.info("Graph Density      : %.6f", graph.statistics.graph_density)

        ##################################################
        # Graph Visualization
        ##################################################

        GraphVisualizer.save(
            patient=patient,
            graph=graph,
            output_dir=f"outputs/graph/{patient.patient_id}",
        )


        ##################################################
        # Patch Group Builder
        ##################################################

        logger.info("")
        logger.info("=" * 80)
        logger.info("PATCH GROUPING")
        logger.info("=" * 80)

        group_builder = PatchGroupBuilder(
            config=config,
            logger=logger,
        )

        patch_groups = group_builder.build(
            graph,
        )

        logger.info(
            "Patch Groups : %d",
            len(patch_groups),
        )

        ##################################################
        # Print Groups
        ##################################################

        for group in patch_groups[:10]:

            logger.info("")
            logger.info(
                "Group %04d",
                group.group_id,
            )

            logger.info(
                "Center Node : %d",
                group.center_node.node_id,
            )

            logger.info(
                "Group Size : %d",
                len(group.member_nodes),
            )

            logger.info(
                "Members : %s",
                [node.node_id for node in group.member_nodes],
            )


        ##################################################
        # Patch Group Serialization
        ##################################################

        logger.info("")
        logger.info("=" * 80)
        logger.info("PATCH GROUP SERIALIZATION")
        logger.info("=" * 80)

        group_serializer = PatchGroupSerializer(
            logger=logger,
        )

        groups_json_path = group_serializer.save(
            patient_id=patient.patient_id,
            patch_groups=patch_groups,
            output_dir=(
               f"outputs/train/{patient.patient_id}"
            ),
        )

        logger.info(
            "Groups saved : %s",
            groups_json_path,
        )



        ##################################################
        # Visualization
        ##################################################

        PatchVisualizer.save(
            patient,
            f"outputs/patch_pipeline_v2/{patient.patient_id}",
        )

       
    logger.info("")
    logger.info("=" * 80)
    logger.info("GRAPH PIPELINE TEST FINISHED")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()