import yaml
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from sam2.build_sam import build_sam2_video_predictor
import torch
import os
from PIL import Image, ImageTk
from tkinter import filedialog
from tkinter import ttk
import tkinter.colorchooser as colorchooser
import sys
import threading
from simple_lama_inpainting import SimpleLama
from tqdm import tqdm

# from diffusers import StableDiffusionInpaintPipeline
from diffusers import DiffusionPipeline
# Load config from YAML
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model_cfg = config['model']['config']
SAM_CHECKPOINTS = config['model']['checkpoints']


class ObjectRemovalApp:
    def __init__(self, root):
        # Create a style object
        self.style = ttk.Style()

        # Customize the slider with a colored track (the 'Horizontal.TScale' is the default theme for horizontal scale)
        self.style.configure("TScale", troughcolor="gray", sliderthickness=20)
        self.root = root
        self.root.title("SAM2 TKinter ")
        self.frames_folder = "./frames/"
        self.video_segments={}
        self.INPAINTING_MODE = "colors"

        self.button_frame = tk.Frame(root)
        self.button_frame.pack(fill=tk.X)

        # self.difussion_settings = {
        #     "prompt": "A person standing in front of a building",
        #     "num_inference_steps": 50,
        #     "guidance_scale": 7.5
        # }

        # Todo: add different colors for different objects
        self.mask_color=(0,255,0)
        

        # Add Play, Pause, and Stop buttons
        self.play_button = tk.Button(self.button_frame, text="Play", command=self.play_video)
        self.play_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.pause_button = tk.Button(self.button_frame, text="Pause", command=self.pause_video)
        self.pause_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.stop_button = tk.Button(self.button_frame, text="Stop", command=self.stop_video)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Add a slider for video progress
        self.progress_slider = ttk.Scale(self.button_frame, from_=0, to=100, orient="horizontal", length=800, command=self.on_slider_move)
        self.progress_slider.pack(side=tk.LEFT, padx=5, pady=5)
    
        
        # Create a menu bar
        self.menubar = tk.Menu(root)
        
        # Create a "File" menu
        file_menu = tk.Menu(self.menubar, tearoff=0)
        file_menu.add_command(label="Open a Video", command=self.load_video)
        file_menu.add_command(label="Open a Folder", command=self.open_folder)
        file_menu.add_command(label="Save Masks", command=self.save_masks)
        file_menu.add_command(label="Save Video", command=self.save_video)
        
        file_menu.add_separator()

        file_menu.add_command(label="Exit", command=root.quit)
        
        # Add "File" menu to the menubar
        self.menubar.add_cascade(label="File", menu=file_menu)

        # Create a "Tools" menu
        tools_menu = tk.Menu(self.menubar, tearoff=0)
        inpainting_menu = tk.Menu(tools_menu, tearoff=0)
        inpainting_menu.add_command(label="Colors", command=self.open_color_settings)
        inpainting_menu.add_command(label="Diffusion", command=self.open_diffusion_settings)
        inpainting_menu.add_command(label="LAMA", command=self.lama_inpainting)
        tools_menu.add_command(label="Process", command=self.process_video)
        tools_menu.add_cascade(label="Inpaint", menu=inpainting_menu)
        tools_menu.add_command(label="Reset", command=self.reset)
        
        # Add "Tools" menu to the menubar
        self.menubar.add_cascade(label="Tools", menu=tools_menu)

        checkpoints_menu = tk.Menu(tools_menu, tearoff=0)
        checkpoints_menu.add_command(label="SAM2", command=None)
        checkpoints_menu.add_command(label="LAMA", command=None)

        # Add "Edit" menu to the menubar
        
        

        # add "help" menu to the menubar
        help_menu=tk.Menu(self.menubar,tearoff=0)
        help_menu.add_command(label="About", command=self.about)
        self.menubar.add_cascade(label="About", menu=help_menu)
        
        # Set the menubar in the root window
        root.config(menu=self.menubar)
        
        # Create a frame to hold both the canvas (for video) and the terminal (text widget)
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(expand=True, fill=tk.BOTH)
        self.canvas = tk.Canvas(self.main_frame, bg='gray')
     
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        
        self.points = []
        self.labels = []  # Store corresponding labels (1 for positive, 0 for negative)
        self.canvas.bind("<Button-1>", lambda event: self.add_point(event, positive=True))  # Left click
        self.canvas.bind("<Button-3>", lambda event: self.add_point(event, positive=False))  # Right click
        
        self.video_path = None

        

        self.current_frame = None
        self.current_frame_idx = 0

        self.frames_list = [] 
        self.processed_frames = []
        self.playing = True
        self.paused = False
        self.frame_rate = 30
        self.total_frames = 0

        self.scale_factor = 1.0
        self.mask = None

    
        # # Terminal output area
        # self.terminal_text = tk.Text(self.main_frame, height=10, bg='black', fg='white', wrap=tk.WORD)
        # self.terminal_text.pack(side=tk.RIGHT, fill=tk.Y)
        
        # # Redirect stdout and stderr to the text widget
        # sys.stdout = TerminalOutput(self.terminal_text)
        # sys.stderr = TerminalOutput(self.terminal_text)

     

        
        # Load SAM model
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.predictor = build_sam2_video_predictor(model_cfg, SAM_CHECKPOINTS, self.device)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load SAM model: {str(e)}")

    def about(self):
        messagebox.showinfo("About", "This is a simple object removal app using the SAM model.")


    def lama_inpainting(self):
        self.processed_frames = []
        simple_lama = SimpleLama()
        
        # Create a progress bar, iterating over video segments with tqdm
        for frame_idx, masks in tqdm(self.video_segments.items(), desc="Inpainting frames", unit="frame"):
            frame_image = Image.open(self.frames_list[frame_idx])
            width, height = frame_image.size
            mask_image = np.zeros((height, width), dtype=np.uint8)
            
            # Create the mask for the current frame
            for obj_id, mask in masks.items():
                mask_image[mask] = 1
                
            mask_image = Image.fromarray((mask_image * 255).astype(np.uint8))
            
            # Convert the mask to a NumPy array for dilation
            mask_np = np.array(mask_image)

            # Define the kernel for dilation
            kernel = np.ones((40, 40), np.uint8)  # Adjust the kernel size as needed
            dilated_mask_np = cv2.dilate(mask_np, kernel, iterations=1)

            # Convert the dilated mask back to a PIL image
            dilated_mask = Image.fromarray(dilated_mask_np)
            
            # Perform inpainting
            generated_image = simple_lama(frame_image, dilated_mask)
            generated_image = np.array(generated_image)
            
            # Append the processed frame
            self.processed_frames.append(generated_image)

        # Show success message after completion
        messagebox.showinfo("Success", "Video inpainted successfully")
            
            
                

            
        
    
    def open_color_settings(self):
        """
        Opens a window to set the color for inpainting
        """
        # Open the color chooser dialog
        selected_color = colorchooser.askcolor(title="Select Inpainting Color")
        
        # The askcolor function returns a tuple with the RGB value and the hexadecimal value of the selected color
        if selected_color[0]:  # If a color is selected (the hex value is not None)
            self.mask_color = np.array(selected_color[0], dtype=np.uint8)  # Convert RGB values to numpy array  # Save the hex value for future use
            messagebox.showinfo("Color Selected", f"Inpainting color selected: {self.mask_color}")
        else:
            messagebox.showwarning("No Color Selected", "No color was selected for inpainting.")
        # Add a confirm button to apply the settings
        self.inpaint_colors()

    def open_diffusion_settings(self):
        """ Opens a window to set parameters for diffusion inpainting """
        diffusion_window = tk.Toplevel(self.root)
        diffusion_window.title("Diffusion Inpainting Settings")

        # Create label and entry for prompt
        prompt_label = tk.Label(diffusion_window, text="Prompt:")
        prompt_label.pack(padx=10, pady=5)
        self.prompt_entry = tk.Entry(diffusion_window)
        self.prompt_entry.pack(padx=10, pady=5)

        # Create label and entry for other settings (e.g., number of steps)
        steps_label = tk.Label(diffusion_window, text="Number of Steps:")
        steps_label.pack(padx=10, pady=5)
        self.steps_entry = tk.Entry(diffusion_window)
        self.steps_entry.pack(padx=10, pady=5)

        # Option to choose strength (from 0.1 to 1.0)
        strength_label = tk.Label(diffusion_window, text="Inpainting Strength:")
        strength_label.pack(padx=10, pady=5)
        self.strength_scale = tk.Scale(diffusion_window, from_=0.1, to=10, orient="horizontal", resolution=0.1, length=200)
        self.strength_scale.pack(padx=10, pady=5)

        # Add a confirm button to apply the settings
        confirm_button = tk.Button(diffusion_window, text="Inpaint!", command=self.inpaint_diffusion)
        confirm_button.pack(padx=10, pady=10)


    

    def inpaint_diffusion(self):
        prompt=self.prompt_entry.get()
        num_inference_steps=int(self.steps_entry.get())
        # guidance_scale=float(self.strength_scale.get())
        guidance_scale=7.5
        num_samples = 3
        self.processed_frames = []
        
        # Load the Stable Diffusion inpainting model
        # pipe = StableDiffusionInpaintPipeline.from_pretrained(
        #     STABLE_DIFFUSION_CHECKPOINTS,
        #     torch_dtype=torch.float16
        # ).to("cuda")

        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16).to("cuda")

        for frame_idx, masks in self.video_segments.items():
            # Convert frame and mask to PIL Image
            frame_image=Image.open(self.frames_list[frame_idx])

            # Get the dimensions (width, height) of the frame image
            width, height = frame_image.size

            # Create an empty mask with the same dimensions as the frame image (height, width)
            mask_image = np.zeros((height, width), dtype=np.uint8)
            # mask_image = np.zeros(frame_image.shape[:2], dtype=np.uint8)
            for obj_id, mask in masks.items():
                mask_image[mask] = 1
            # Invert the mask: 1 for areas to inpaint, 0 for areas to preserve
            # inverted_mask = 1 - mask
            mask_image = Image.fromarray((mask * 255).astype(np.uint8))

            # save the mask_image numpy array and frame_image
            generator = torch.Generator(device="cuda").manual_seed(0) # change the seed to get different results
            
            # mask_image.save("mask_image.png")
            # frame_image_arr= np.array(frame_image)
            # cv2.imwrite("frame_image.png", cv2.cvtColor(frame_image_arr, cv2.COLOR_RGB2BGR))
            # Inpaint the frame
            inpainted_image = pipe(
                prompt=prompt,
                image=frame_image,
                mask_image=mask_image,
                guidance_scale=guidance_scale,
                generator=generator,
                num_images_per_prompt=num_samples,
                num_inference_steps=num_inference_steps
            ).images[0]

            # Convert inpainted image back to numpy array
            inpainted_array = np.array(inpainted_image)
            # Save the inpainted frame
            # cv2.imwrite("inpainted_frame.png", cv2.cvtColor(inpainted_array, cv2.COLOR_RGB2BGR))
            cv2.resize(inpainted_array, (height, width))
            # exit(0)

            self.processed_frames.append(inpainted_array)

        messagebox.showinfo("Success", f"Video inpainted successfully")
           

    

    def open_folder(self):
        folder_selected = filedialog.askdirectory(title="Select Folder")
        if folder_selected:  # If a folder is selected
            self.frames_folder = folder_selected
            print(f"Selected folder: {self.frames_folder}")
        self.current_frame = None
        self.frames_list = sorted([os.path.join(self.frames_folder, f) for f in os.listdir(self.frames_folder)
                                   if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.total_frames = len(self.frames_list)
        self.progress_slider.config(to=self.total_frames-1)
        for file in os.listdir(self.frames_folder):
            if file.endswith(".jpg") or file.endswith(".png"):
                frame = cv2.imread(os.path.join(self.frames_folder, file))
                self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.current_frame_idx = 0
                break
        if self.current_frame is not None:
            self.display_frame()
            self.inference_state = self.predictor.init_state(video_path=self.frames_folder)
        else:
            messagebox.showerror("Error", "No image files found in the selected folder.")

    def on_slider_move(self, value):
        """Handle slider movement and update video progress."""
    
        self.current_frame_idx = int(float(value))
        self.current_frame = cv2.imread(self.frames_list[self.current_frame_idx])
        self.current_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
    
        # self.show_frame()  # Update the displayed frame when slider is moved

    def play_video(self):
        self.playing = True
        self.playing = True
        self.paused = False
        self.show_frame()

    def pause_video(self):
        self.paused = True


    def stop_video(self):
        self.playing = False
        self.paused = False

        self.progress_slider.set(0)  # Reset slider

    def show_frame(self):
        show_processed_frame= True if len(self.processed_frames)>0 else False
        if not self.playing or self.paused or not self.frames_list:
            print("Not playing or there is no Frame List", len(self.frames_list))
            return
        if show_processed_frame:
            frame = self.processed_frames[self.current_frame_idx]
        else:
            frame_path = self.frames_list[self.current_frame_idx]

            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the frame to RGB

        # Convert frame to ImageTk format
        img = Image.fromarray(frame)
        img = self.resize_image(img)
        imgtk = ImageTk.PhotoImage(image=img)

        # Display the image on the canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.photo = imgtk  # Keep a reference to avoid garbage collection

        # Update the slider position
        self.progress_slider.set(self.current_frame_idx)

        # slider_x = self.progress_slider.winfo_rootx() - self.root.winfo_rootx()  # X position relative to the root window
        # slider_y = self.progress_slider.winfo_rooty() - self.root.winfo_rooty() -20 # Y position relative to the root window
        # slider_width = self.progress_slider.winfo_width()  # Slider width
        # slider_height = self.progress_slider.winfo_height()  # Slider height
        
        # # Calculate the progress ratio
        # progress_ratio = self.current_frame_idx / self.total_frames  # Fraction of video played
        # red_width = int(progress_ratio * slider_width)  # Width of the red rectangle

        # # Clear any previously drawn red overlay
        # self.canvas.delete("played_overlay")

        # # Draw the red overlay (progress bar) directly on top of the slider
        # self.canvas.create_rectangle(slider_x, slider_y, slider_x + red_width, slider_y + slider_height, 
        #                             fill='red', tags="played_overlay")
        # Call show_frame() again after a delay to update the next frame
        self.current_frame_idx += 1
        if self.current_frame_idx >= self.total_frames:
            self.current_frame_idx = 0  # Loop back to the first frame

        self.root.after(int(1000 / self.frame_rate), self.show_frame)
        
    def save_masks(self):
        output_folder=filedialog.askdirectory(title="Select Folder")
        print(f"Selected folder: {output_folder}")
        if not output_folder:
            return
        for frame_idx, masks in self.video_segments.items():
            for obj_id, mask in masks.items():
                mask_filename = os.path.join(output_folder, f"{frame_idx:04d}_{obj_id:02d}.png")
                cv2.imwrite(mask_filename, (mask * 255).astype(np.uint8))
        messagebox.showinfo("Success", f"Segmentation masks saved successfully to {output_folder}")

    
    def save_video(self):
        # Open a save file dialog to select the output path
        output_path = filedialog.asksaveasfilename(
            title="Save Video",
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )

        if not output_path:
            return  # User canceled, so exit the function

        try:
            # Open the input video
            cap = cv2.VideoCapture(self.video_path)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
            if len(self.processed_frames)==0:
                frame_idx = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Draw mask on the current frame, if it exists
                    if frame_idx in self.video_segments:
                        mask = next(iter(self.video_segments[frame_idx].values()))
                        frame[mask] = [0, 255, 0] 

                    out.write(frame)
                    frame_idx += 1
                    self.processed_frames.append(frame)
            else:
                for frame in self.processed_frames:
                    frame=cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame)

            cap.release()
            out.release()

            messagebox.showinfo("Success", f"Video saved successfully to {output_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save video: {str(e)}")


    def load_video(self):
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        ]
        self.video_path = filedialog.askopenfilename(filetypes=filetypes)
        if self.video_path:
            if not os.path.exists(self.video_path):
                messagebox.showerror("Error", f"The selected file does not exist: {self.video_path}")
                return
            
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                messagebox.showerror("Error", f"Unable to open video file: {self.video_path}")
                return

            # Create 'frames' folder if it doesn't exist
            self.frames_folder = "./frames/"
            if not os.path.exists(self.frames_folder):
                os.makedirs(self.frames_folder)

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Save frame as JPG in the frames folder
                frame_filename = os.path.join(self.frames_folder, f"{frame_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                frame_count += 1

            cap.release()
            self.inference_state= self.predictor.init_state(video_path="./frames/")

            # Load the first frame for display
            if frame_count > 0:
                self.total_frames = frame_count
                self.progress_slider.config(to=self.total_frames-1)
                self.frames_list = sorted([os.path.join(self.frames_folder, f) for f in os.listdir(self.frames_folder)
                                   if f.endswith(('.png', '.jpg', '.jpeg'))])
                first_frame=0
                frame_filename = os.path.join(self.frames_folder, f"{first_frame:04d}.jpg")
                frame=cv2.imread(frame_filename)
                self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.display_frame()
            else:
                messagebox.showerror("Error", "Failed to extract frames from the video.")
        else:
            messagebox.showinfo("Info", "No file selected.")

    def resize_image(self, image):
        window_width = self.root.winfo_width()
        window_height = self.root.winfo_height()
        img_width, img_height = image.size
        
        # Calculate the scaling factor
        width_ratio = window_width / img_width
        height_ratio = window_height / img_height
        self.scale_factor = min(width_ratio, height_ratio, 1.0)
        
        new_width = int(img_width * self.scale_factor)
        new_height = int(img_height * self.scale_factor)
        
        return image.resize((new_width, new_height), Image.LANCZOS)
    
    

    def display_frame(self):
        if self.current_frame is not None:
            image = Image.fromarray(self.current_frame)
            resized_image = self.resize_image(image)
            self.photo = ImageTk.PhotoImage(resized_image)
            
            # Update canvas size
            self.canvas.config(width=self.photo.width(), height=self.photo.height())
            
            self.canvas.delete("all")  # Clear previous content
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            self.draw_points()

    def add_point(self, event, positive=True):
        x, y = event.x / self.scale_factor, event.y / self.scale_factor
        self.points.append((x, y))
        self.labels.append(1 if positive else 0)  # Label 1 for positive, 0 for negative
        self.draw_points()
        self.update_mask()

    def remove_point(self, event):
        x, y = event.x / self.scale_factor, event.y / self.scale_factor
        for i, (px, py) in enumerate(self.points):
            if abs(x - px) < 5 and abs(y - py) < 5:
                self.points.pop(i)
                self.labels.pop(i)
                self.draw_points()
                self.update_mask()
                break

    def inpaint_colors(self):
        self.processed_frames=[]
        for frame_idx, masks in self.video_segments.items():
            frame = cv2.imread(self.frames_list[frame_idx])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for obj_id, mask in masks.items():
                frame[mask] = self.mask_color
                self.processed_frames.append(frame)
        messagebox.showinfo("Success", f"Colors inpainted successfully")

    def draw_points(self):
        self.canvas.delete("point")  # Remove previously drawn points
        for i, (x, y) in enumerate(self.points):
            scaled_x, scaled_y = x * self.scale_factor, y * self.scale_factor
            color = "red" if self.labels[i] == 1 else "blue"  # Red for positive, Blue for negative
            self.canvas.create_oval(scaled_x-2, scaled_y-2, scaled_x+2, scaled_y+2, fill=color, tags="point")

    def update_mask(self):
        if not self.points or self.current_frame is None:
            return
        ann_frame_idx = 0
        ann_obj_id = 0
        input_points = np.array(self.points)
        input_labels = np.array(self.labels)
        
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=input_points,
            labels=input_labels,
        )
        self.mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()
        
        # Apply the mask to the frame
        masked_frame = self.current_frame.copy()
        masked_frame[self.mask] = self.mask_color
        
        # Display the masked frame
        image = Image.fromarray(masked_frame.astype('uint8'))
        resized_image = self.resize_image(image)
        self.photo = ImageTk.PhotoImage(resized_image)
        
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.draw_points()

    def process_video(self):
        if len(self.frames_list)==0 or not self.points:
            messagebox.showwarning("Warning", "Please load a video and select points first.")
            return

        if self.mask is None:
            messagebox.showwarning("Warning", "Please generate a mask by adding points before processing.")
            return

        try:
            # run propagation throughout the video and collect the results in a dict
            self.video_segments = {}  # video_segments contains the per-frame segmentation results
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
                self.video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            self.inpaint_colors()

            messagebox.showinfo("Success", f"Video processed successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process video: {str(e)}")

    def reset(self):
        self.points = []
        self.labels = []
        self.processed_frames = []
        self.mask = None
        self.canvas.delete("all")  # Clear the canvas
        if self.current_frame is not None:
            self.display_frame()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1280x720")  # Set initial window size
    app = ObjectRemovalApp(root)
    root.mainloop()
