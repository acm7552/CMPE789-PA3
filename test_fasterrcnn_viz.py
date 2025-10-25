import os
import torch
import argparse
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import cv2
import numpy as np

# import data
from MOT16 import MOT16, detection_collate

tf = transforms.ToTensor()

def load_data():
    testMOT16data = MOT16(root = 'MOT16/test', transform=tf, getGroundTruth=False, useDetections=True)

    test_loader = DataLoader(
        testMOT16data, batch_size=16, shuffle=False,
        pin_memory=True,
        collate_fn=detection_collate, num_workers=2, persistent_workers=True
    )
    return test_loader


def load_model(weights_pth, device):
    # no weights yet, gonna load them in
    model = fasterrcnn_resnet50_fpn(weights=None)
    #in_features = model.roi_heads.box_predictor.cls_score.in_features
    #model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)  # mot16 only does people or background

    # load the state dict from fine tuning
    loadWeights = torch.load(weights_pth, map_location=device, weights_only=True)
    model.load_state_dict(loadWeights)
    model.to(device)
    model.eval()
    return model

def load_pretrained(device):

    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model.to(device)
    model.eval()
    return model


def get_detections(model, image, device, thresh, save=True):
    img = Image.open(image).convert("RGB")
    img_tensor = tf(img).to(device)

    # make predictions
    with torch.no_grad():
        predictions = model([img_tensor])

    pred = predictions[0]
    boxes = pred["boxes"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()

    # filter by confidence
    keep = scores >= thresh
    boxes = boxes[keep]
    scores = scores[keep]

    # show detections
    if save:
        img_cv = np.array(img)[:, :, ::-1].copy()
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_cv, f"{score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        out_path = os.path.join("outputs", os.path.basename(image))
        os.makedirs("outputs", exist_ok=True)
        #cv2.imwrite(out_path, img_cv)
        #print(f"saved detections to {out_path}")

    return boxes, scores, img_cv

def play_sequence(model, sequence_dir, device, thresh=0.3, save_path=None, vidName=None):
    # get frames
    frames = sorted([os.path.join(sequence_dir, f) for f in os.listdir(sequence_dir) if f.endswith(".jpg")])

    video = None
    if save_path:

        os.makedirs(save_path, exist_ok=True)
        # get frame size from first image
        sample = cv2.imread(frames[0])
        h, w = sample.shape[:2]

        video_file = os.path.join(save_path, f"{vidName}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
        out_vid = cv2.VideoWriter(video_file, fourcc, 30, (w, h))

    for frame_path in frames:
        # get_detections returns boxes, scores, but you need the image with rectangles
        boxes, scores, img_cv = get_detections(model, frame_path, device, thresh, save=True)

        # display
        cv2.imshow("Detections", img_cv)
        if out_vid:
            out_vid.write(img_cv)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    if out_vid:
        out_vid.release()
        print(f"Saved video to {video_file}")



def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to model params")
    ap.add_argument("--sequence", required=True, help="path to frames for tracking, i.e. 'MOT16/test/MOT16-01/'")
    #ap.add_argument("--vidname", required=True, help="name of output video")
    args = ap.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #testFrame = "MOT16/test/MOT16-01/img1/000001.jpg"

    testSequence = f"{args.sequence}img1/"

    parent = os.path.basename(os.path.dirname(testSequence))  # "MOT16-01"
    model_name = args.model.replace(".pth", "")

    video_name = f"{parent}_{model_name}"
    if args.model.lower() == 'pretrained':
        pretrained = load_pretrained(device)
        play_sequence(pretrained, testSequence, device, thresh=0.5, save_path="videos/", vidName = video_name)
        return
    else:
        finetunedmodel = load_model("finetunedfasterrcnn_old.pth", device)
        #get_detections(finetunedmodel, testFrame, device, thresh=0.5, save=True)
        play_sequence(finetunedmodel, testSequence, device, thresh=0.5, save_path="videos/", vidName = video_name)



if __name__ == "__main__":

    main()
