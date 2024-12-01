import re
import cv2
import easyocr
import numpy as np
import pandas as pd

from deskew import determine_skew
from skimage.transform import rotate
from ultralytics import YOLO


class AudiogramReader:
    def __init__(self, box_model_path: str, symbol_model_path: str):
        self._box_model = YOLO(box_model_path)
        self._symbol_model = YOLO(symbol_model_path)
        self._reader = easyocr.Reader(["en"], gpu=True)

    def feature_extraction(self, image: np.ndarray):
        image = self._load_image(image)

        graph, _ = self._crop_graph_table(image)

        air_rt_df, air_lt_df, bone_rt_df, bone_lt_df = self._detect_symbols(graph)
        air_rt_df, air_lt_df, bone_rt_df, bone_lt_df = self._select_symbols(
            air_rt_df, air_lt_df, bone_rt_df, bone_lt_df
        )

        return air_rt_df["y"], air_lt_df["y"]

    def _load_image(self, image: np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, _ = self._strighten_image(image)
        return image

    def _strighten_image(self, image: np.ndarray):
        angle = determine_skew(image)
        rotated = rotate(image, angle, resize=True) * 255
        rotated_img = rotated.astype(np.uint8)
        return rotated_img, angle

    def _crop_graph_table(self, image, margin: int = 20):
        box_results = self._box_model.predict(image)
        x1, y1, x2, y2 = self._get_positions(box_results, 0).astype(int)[0]  # [x1, y1, x2, y2]
        graph = image[y1 - margin : y2 + margin, x1 - margin : x2 + margin]
        x1, y1, x2, y2 = self._get_positions(box_results, 1).astype(int)[0]  # [x1, y1, x2, y2]
        table = image[y1 - margin : y2 + margin, x1 - margin : x2 + margin]
        table = cv2.resize(table, (431, 1023), interpolation=cv2.INTER_AREA)
        return graph, table

    def _get_positions(self, results, label):
        cls = results[0].boxes.cls.cpu().numpy()
        index = np.where(cls == label)[0]
        positions = results[0].boxes.xyxy[index].cpu().numpy()  # [x1, y1, x2, y2]
        return positions

    def _extract_text_from_table(self, table):
        acrt = table[75:200, 220:400]
        aclt = table[190:325, 220:400]
        bcrt = table[300:425, 220:400]
        bclt = table[415:545, 220:400]

        srtrt = table[680:850, 110:260]
        slrt = table[680:760, 220:320]
        pbrt = table[760:850, 220:360]

        srtlt = table[840:1000, 110:260]
        sllt = table[840:920, 220:320]
        pblt = table[920:1000, 220:360]

        acrt_text = self._get_text_from_image(acrt)
        aclt_text = self._get_text_from_image(aclt)
        bcrt_text = self._get_text_from_image(bcrt)
        bclt_text = self._get_text_from_image(bclt)

        srtrt_text = self._get_text_from_image(srtrt)
        slrt_text = self._get_text_from_image(slrt)
        pbrt_text = self._get_text_from_image(pbrt)

        srtlt_text = self._get_text_from_image(srtlt)
        sllt_text = self._get_text_from_image(sllt)
        pblt_text = self._get_text_from_image(pblt)

        return (
            acrt_text,
            aclt_text,
            bcrt_text,
            bclt_text,
            srtrt_text,
            slrt_text,
            pbrt_text,
            srtlt_text,
            sllt_text,
            pblt_text,
        )

    def _get_text_from_image(self, image, transform: bool = True):
        results = self._reader.readtext(image, allowlist="0123456789")
        if len(results) == 0:
            text = ""
        else:
            text = results[0][1]
        if transform:
            match = re.search(r"\d+", text)
            if match:
                text = int(match.group())
        return text

    def _detect_symbols(self, graph):
        """
        class_dict = {0: 'Air Rt Unmasked', 1: 'Air Lt Unmasked', 2: 'Air Rt Masked',
                3: 'Air Lt masked', 4: 'Bone Rt Unmasked', 5: 'Bone Lt Unmasked',
                6: 'Bone Rt Masked', 7: 'Bone Lt Masked'}
        """
        symbol_results = self._symbol_model(graph)

        air_rt_um = pd.DataFrame(self._predict_db(graph, symbol_results, 0, 20))
        air_lt_um = pd.DataFrame(self._predict_db(graph, symbol_results, 1, 20))
        air_rt_m = pd.DataFrame(self._predict_db(graph, symbol_results, 2, 20))
        air_lt_m = pd.DataFrame(self._predict_db(graph, symbol_results, 3, 20))
        bone_rt_um = pd.DataFrame(self._predict_db(graph, symbol_results, 4, 20))
        bone_lt_um = pd.DataFrame(self._predict_db(graph, symbol_results, 5, 20))
        bone_rt_m = pd.DataFrame(self._predict_db(graph, symbol_results, 6, 20))
        bone_lt_m = pd.DataFrame(self._predict_db(graph, symbol_results, 7, 20))

        air_rt_df = (
            pd.concat([air_rt_um, air_rt_m], axis=0)
            .sort_values(by=0)
            .rename(columns={0: "x", 1: "y"})
        )
        air_lt_df = (
            pd.concat([air_lt_um, air_lt_m], axis=0)
            .sort_values(by=0)
            .rename(columns={0: "x", 1: "y"})
        )
        bone_rt_df = (
            pd.concat([bone_rt_um, bone_rt_m], axis=0)
            .sort_values(by=0)
            .rename(columns={0: "x", 1: "y"})
        )
        bone_lt_df = (
            pd.concat([bone_lt_um, bone_lt_m], axis=0)
            .sort_values(by=0)
            .rename(columns={0: "x", 1: "y"})
        )

        return air_rt_df, air_lt_df, bone_rt_df, bone_lt_df  # [x, y]

    def _predict_db(self, graph, results, label, margin):
        image = graph.copy()
        image = image[margin:-margin, margin:-margin]

        positions = self._get_positions(results, label)
        positions = self._shift_positions(positions, margin)
        positions = self._scale_positions(positions, image.shape[1], image.shape[0])

        centers = self._get_centers(positions)  # [x,y]
        centers = np.array(sorted(centers, key=lambda x: x[0]))

        if centers.shape[0] == 0:
            return [np.NaN]

        return centers

    def _shift_positions(self, position, margin):
        position[:, 0] = position[:, 0] - margin
        position[:, 1] = position[:, 1] - margin
        position[:, 2] = position[:, 2] - margin
        position[:, 3] = position[:, 3] - margin
        return position

    def _scale_positions(self, positions, image_width, image_height):
        positions[:, [0, 2]] = (positions[:, [0, 2]] / image_width) * 6  # shift x
        positions[:, [1, 3]] = ((positions[:, [1, 3]] / image_height) * 130) - 10  # shift y
        return positions

    def _get_centers(self, positions):
        centers = (positions[:, [0, 1]] + positions[:, [2, 3]]) / 2  # x, y
        return centers

    def _select_symbols(self, air_rt_df, air_lt_df, bone_rt_df, bone_lt_df):
        air_rt_df = self._select_symbol(air_rt_df, "air").reset_index(drop=True)
        air_rt_df = air_rt_df.rename(
            index={
                0: "250",
                1: "500",
                2: "1000",
                3: "2000",
                4: "4000",
                5: "6000",
                6: "8000",
            }
        )

        air_lt_df = self._select_symbol(air_lt_df, "air").reset_index(drop=True)
        air_lt_df = air_lt_df.rename(
            index={
                0: "250",
                1: "500",
                2: "1000",
                3: "2000",
                4: "4000",
                5: "6000",
                6: "8000",
            }
        )

        bone_rt_df = self._select_symbol(bone_rt_df, "bone").reset_index(drop=True)
        bone_rt_df = bone_rt_df.rename(index={0: "500", 1: "1000", 2: "2000", 3: "4000"})

        bone_lt_df = self._select_symbol(bone_lt_df, "bone").reset_index(drop=True)
        bone_lt_df = bone_lt_df.rename(index={0: "500", 1: "1000", 2: "2000", 3: "4000"})

        return air_rt_df, air_lt_df, bone_rt_df, bone_lt_df

    def _select_symbol(self, df, c_type: str):
        if c_type == "air":
            x_map = [1, 2, 3, 4, 5, 5.5, 6]
        if c_type == "bone":
            x_map = [2, 3, 4, 5]

        df = df.dropna()
        xs = df["x"].values

        if xs.shape[0] == 0:
            return pd.DataFrame([[np.NaN] * 2] * (len(x_map)), columns=["x", "y"])

        x_needs = [(np.abs(xs - i)).argmin() for i in x_map]

        return df.iloc[x_needs]
