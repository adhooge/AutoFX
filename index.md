# ATIAM Internship Summer 2022
## Lab: [Sony CSL](https://github.com/SonyCSLParis)
__Supervisor:__ [Gaëtan Hadjeres](https://github.com/Ghadjeres)
<div align="center">
<img src="https://adhooge.github.io/AutoFX/logos/sorbonne.jpg" align="left" alt="Logo Sorbonnes universités" width="200">
<img src="https://adhooge.github.io/AutoFX/logos/telecom.jpg" alt="Logo Télécom Paris" width="100">
<img src="https://adhooge.github.io/AutoFX/logos/ircam.jpg" align="right" alt="Logo IRCAM" width="250">
</div>
<div align="center">
<img src="https://adhooge.github.io/AutoFX/logos/atiam.jpg" align="left" alt="Logo Sorbonnes universités" width="200">
<img src="https://adhooge.github.io/AutoFX/logos/csl.png" align="right" alt="Logo Sony CSL" width="250">
</div>
<br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>

### Abstract

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed non risus. Suspendisse lectus tortor, dignissim sit amet, adipiscing nec, ultricies sed, dolor. Cras elementum ultrices diam. Maecenas ligula massa, varius a, semper congue, euismod non, mi. Proin porttitor, orci nec nonummy molestie, enim est eleifend mi, non fermentum diam nisl sit amet erat. Duis semper. Duis arcu massa, scelerisque vitae, consequat in, pretium a, enim. Pellentesque congue. Ut in risus volutpat libero pharetra tempor. Cras vestibulum bibendum augue. Praesent egestas leo in pede. Praesent blandit odio eu enim. Pellentesque sed dui ut augue blandit sodales. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; Aliquam nibh. Mauris ac mauris sed pede pellentesque fermentum. Maecenas adipiscing ante non diam sodales hendrerit.


### Dataset

This work is based on the __Audio Effects__ Dataset released by *Fraunhofer IDMT* in 2010. It is available [here](https://www.idmt.fraunhofer.de/en/publications/datasets/audio_effects.html)
and has been created for the purpose of the classification of audio effects on polyphonic and monophonic bass and guitar recordings.
See the [published paper](https://www.aes.org/e-lib/browse.cfm?elib=15310) for more information:
>Stein, Michael, et al. "Automatic detection of audio effects in guitar and bass recordings." Audio Engineering Society Convention 128. Audio Engineering Society, 2010.

#### Example sounds from the dataset

<table>
<caption><b> Dataset examples</b></caption>
    <tr>
    <td style="text-align: center; vertical-align: middle;"><b>Clean</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>Feedback Delay</b></td>
</tr>
<tr>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/idmt/dry.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/idmt/fdback_delay.wav">
</audio>
</td>
</tr>
<tr>
    <td style="text-align: center; vertical-align: middle;"><b>Slapback Delay</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>Reverb</b></td></tr>
<tr><td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/idmt/slpback_delay.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/idmt/reverb.wav">
</audio>
</td></tr><tr>
    <td style="text-align: center; vertical-align: middle;"><b>Chorus</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>Flanger</b></td></tr>
<tr><td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/idmt/chorus.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/idmt/flanger.wav">
</audio>
</td></tr><tr>
    <td style="text-align: center; vertical-align: middle;"><b>Phaser</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>Tremolo</b></td></tr>
<tr><td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/idmt/phaser.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/idmt/tremolo.wav">
</audio>
</td></tr><tr>
    <td style="text-align: center; vertical-align: middle;"><b>Vibrato</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>Overdrive</b></td>
</tr><tr>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/idmt/vibrato.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/idmt/overdrive.wav">
</audio>
</td></tr>
<tr>
<td style="text-align: center; vertical-align: middle;"><b>Distortion</b></td>
</tr>
<tr>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/idmt/distortion.wav">
</audio>
</td>
</tr>
</table>

### Synthetic sounds

A synthetic dataset is obtained by processing clean audio from the IDMT Dataset with the [`pedalboard`](https://github.com/spotify/pedalboard) library.

- **Modulation** sounds are obtained with the `pedalboard.Chorus` plugin;
- **Delay** sounds with the `pedalboard.Delay` plugin;
- **Distortion** sounds with an effect chain of `pedalboard.Distortion`, `pedalboard.LowShelfFilter` and `pedalboard.HighShelfFilter`.

Example sounds for each of those are available below:

<table>
<caption><b> Synthetic sound examples</b></caption>
    <tr>
    <td style="text-align: center; vertical-align: middle;"><b>Modulation</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>Delay</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>Distortion</b></td>
</tr>
<tr>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/synth/0-Modulation.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/synth/0-Delay.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/synth/0-Distortion.wav">
</audio>
</td>
</tr>
<tr>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/synth/1-Modulation.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/synth/1-Delay.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/synth/1-Distortion.wav">
</audio>
</td>
</tr>
<tr>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/synth/2-Modulation.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/synth/2-Delay.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/synth/2-Distortion.wav">
</audio>
</td>
</tr>
<tr>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/synth/3-Modulation.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/synth/3-Delay.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/synth/3-Distortion.wav">
</audio>
</td>
</tr>
<tr>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/synth/4-Modulation.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/synth/4-Delay.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/synth/4-Distortion.wav">
</audio>
</td>
</tr>
</table>