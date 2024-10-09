from justinsdeeplearningutilities.yolo_tracking import track_video




if __name__ == '__main__':

    input_video_path = "./italy_test_video.mp4"

    output_video_path = track_video(input_video_path)

    print(f"Processed video saved to: {output_video_path}")


