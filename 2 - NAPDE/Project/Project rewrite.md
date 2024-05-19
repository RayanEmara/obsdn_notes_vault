I have two files:
- `broadband.pt`
- `lowpass.pt`

These files contain tuples $(\mathrm{~n}_{records} , 3, 6000)$ where $\mathrm{~n}_{records}$ is just an index identifying a single record. 
So if I were to go to go to $(0,0,:)$ in `broadband.pt` files I'd find an array containing 6 seconds of (time-ordered) amplitudes. `lowpass.pt` would contain the same thing except it would be the low-passed version.  
The second dimension is a spatial dimension, there are three of them for each record `(0,1,2)`.
I want to create and train a model that can generate broadband samples given a lowpass sample.
