import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import json
import subprocess
import threading
import queue
import webbrowser

class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Generative Diffusion Model for Network Optimization")
        self.root.geometry("1200x700")

        self.main_notebook = ttk.Notebook(self.root)
        self.main_notebook.pack(fill=tk.BOTH, expand=True)

        self.create_parameter_tab()
        self.create_output_tab()

        self.output_queue = queue.Queue()
        self.root.after(100, self.check_output_queue)

        self.training_thread = None
        self.tensorboard_process = None
        self.stop_training_flag = threading.Event()

    def create_parameter_tab(self):
        main_frame = ttk.Frame(self.main_notebook)
        self.main_notebook.add(main_frame, text='Parameters and Training')

        param_frame = ttk.Frame(main_frame)
        param_frame.grid(row=0, column=0, sticky="nsew")
        self.output_frame = ttk.Frame(main_frame)
        self.output_frame.grid(row=0, column=1, sticky="nsew")
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        self.common_variables = {
            'exploration_noise': tk.DoubleVar(value=0.1),
            'step_per_epoch': tk.IntVar(value=10),
            'step_per_collect': tk.IntVar(value=10),
            'seed': tk.IntVar(value=1),
            'buffer_size': tk.DoubleVar(value=1e6),
            'epoch': tk.IntVar(value=1000000),
            'batch_size': tk.IntVar(value=512),
            'actor_lr': tk.DoubleVar(value=1e-4),
            'critic_lr': tk.DoubleVar(value=1e-4),
            'n_timesteps': tk.IntVar(value=6),
            'beta_schedule': tk.StringVar(value='vp'),
            'device': tk.StringVar(value='cpu'),
            'bc_coef': tk.BooleanVar(value=False),
            'prior_alpha': tk.DoubleVar(value=0.4),
            'prior_beta': tk.DoubleVar(value=0.4),
        }

        self.advanced_variables = {
            'algorithm': tk.StringVar(value="diffusion_opt"),
            'tau': tk.DoubleVar(value=0.005),
            'wd': tk.DoubleVar(value=1e-4),
            'gamma': tk.DoubleVar(value=1),
            'n_step': tk.IntVar(value=3),
            'logdir': tk.StringVar(value='log'),
            'training_num': tk.IntVar(value=1),
            'test_num': tk.IntVar(value=1),
            'log_prefix': tk.StringVar(value='default'),
            'render': tk.DoubleVar(value=0.1),
            'rew_norm': tk.IntVar(value=0),
            'resume_path': tk.StringVar(value=''),
            'watch': tk.BooleanVar(value=False),
            'prioritized_replay': tk.BooleanVar(value=False),
            'lr_decay': tk.BooleanVar(value=False),
            'note': tk.StringVar(value=''),
        }

        param_notebook = ttk.Notebook(param_frame)
        param_notebook.pack(fill=tk.BOTH, expand=True)

        common_frame = ttk.Frame(param_notebook)
        param_notebook.add(common_frame, text='Common')

        advanced_frame = ttk.Frame(param_notebook)
        param_notebook.add(advanced_frame, text='Advanced')

        self.create_param_widgets(common_frame, self.common_variables)
        self.create_param_widgets(advanced_frame, self.advanced_variables)

        button_frame = ttk.Frame(param_frame)
        button_frame.pack(pady=10)

        ttk.Button(button_frame, text="Submit and Start", command=self.on_submit).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Stop Training", command=self.stop_training).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Start TensorBoard", command=self.start_tensorboard).pack(side=tk.LEFT, padx=5)

    def create_param_widgets(self, parent, variables):
        for i, (key, var) in enumerate(variables.items()):
            ttk.Label(parent, text=key.replace('_', ' ').title()).grid(column=0, row=i, sticky=tk.W)
            if isinstance(var, tk.BooleanVar):
                ttk.Checkbutton(parent, variable=var).grid(column=1, row=i, sticky=tk.W)
            elif key == 'beta_schedule':
                ttk.Combobox(parent, textvariable=var, values=["vp", "linear", "cosine"]).grid(column=1, row=i, sticky=(tk.W, tk.E))
            elif key in ['prior_alpha', 'prior_beta']:
                ttk.Scale(parent, from_=0, to=1, orient=tk.HORIZONTAL, variable=var).grid(column=1, row=i, sticky=(tk.W, tk.E))
                ttk.Label(parent, textvariable=var).grid(column=2, row=i, sticky=tk.W)
            else:
                ttk.Entry(parent, textvariable=var).grid(column=1, row=i, sticky=(tk.W, tk.E))

    def create_output_tab(self):
        self.output_text = scrolledtext.ScrolledText(self.output_frame, wrap=tk.WORD, height=20)
        self.output_text.pack(fill=tk.BOTH, expand=True)

    def on_submit(self):
        params = {**self.common_variables, **self.advanced_variables}
        args = {key: var.get() for key, var in params.items()}
        self.start_training(args)
        return args

    def start_training(self, args):
        if self.training_thread and self.training_thread.is_alive():
            messagebox.showinfo("Training", "Training is already running.")
            return

        self.stop_training_flag.clear()

        def run_training():
            from main import main
            main(args, self.update_output, self.should_stop_training)

        self.training_thread = threading.Thread(target=run_training)
        self.training_thread.start()

    def should_stop_training(self):
        return self.stop_training_flag.is_set()

    def stop_training(self):
        if self.training_thread and self.training_thread.is_alive():
            self.stop_training_flag.set()
            self.update_output("Training stopped.")
        else:
            self.update_output("No training process to stop.")

    def update_output(self, message):
        self.output_queue.put(message)
        self.root.update_idletasks()  # Force GUI update

    def check_output_queue(self):
        while not self.output_queue.empty():
            message = self.output_queue.get()
            self.output_text.insert(tk.END, message + '\n')
            self.output_text.see(tk.END)
        self.root.after(100, self.check_output_queue)

    def start_tensorboard(self):
        logdir = self.advanced_variables['logdir'].get()  # Access logdir from the correct dictionary
        if not self.tensorboard_process or self.tensorboard_process.poll() is not None:
            def run_tensorboard():
                try:
                    self.tensorboard_process = subprocess.Popen(
                        ['tensorboard', '--logdir', logdir, '--port', '6006'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True)
                    webbrowser.open('http://localhost:6006')
                except Exception as e:
                    self.update_output(f"Failed to start TensorBoard: {str(e)}")

            threading.Thread(target=run_tensorboard, daemon=True).start()
        else:
            webbrowser.open('http://localhost:6006')
            messagebox.showinfo("TensorBoard", "TensorBoard is already running.")

def create_gui():
    gui = GUI()
    return gui.root, gui.start_training, gui.update_output, gui.stop_training

if __name__ == '__main__':
    root, on_submit, update_output, stop_training = create_gui()
    root.mainloop()
    args = on_submit()
    print(json.dumps(args, indent=2))
