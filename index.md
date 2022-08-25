# ATIAM Internship Summer 2022
## Lab: [Sony CSL](https://csl.sony.fr/)
__Supervisor:__ [Gaëtan Hadjeres](https://github.com/Ghadjeres)
[Report](https://adhooge.github.io/AutoFX/intership_report.pdf)
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

Many musicians use audio effects to shape their sound to the point that these effects become part of their sound identity. However, configuring audio effects often requires expert knowledge to find the correct settings to reach the desired sound. In this report, we present a novel method to automatically recognize the effect present in a reference sound and find parameters that allow to reproduce its timbre. This tool aims at helping artists during their creative process to quickly configure effects to reproduce a chosen sound, making it easier to explore similar sounds afterwards, similarly to what presets offer but in a much more flexible manner.\\
We implement a classification algorithm to recognize the audio effect used on a guitar sound and reach up to 95 \% accuracy on the test set. This classifier is also compiled and can be used as a standalone plugin to analyze a sound and automatically instanciate the correct effect. We also propose a pipeline to generate a synthetic dataset of guitar sounds processed with randomly configured effects at speeds unreachable before. We use that dataset to train different neural networks to automatically retrieve effect's parameters. We demonstrate that a feature-based approach with typical Music Information Retrieval (MIR) features compete with a larger Convolutional Neural Network (CNN) trained on audio spectrograms while being faster to train and requiring far less parameters. Contrary to the existing literature, making the effects we use differentiable does not allow to improve the performance of our networks which already propose fair reproduction of unseen audio effects when trained only in a supervised manner on a loss on the parameters. We further complete our results with an online perceptual experiment that shows that the proposed approach yields sound matches that are much better than using random parameters, suggesting that this technique is indeed promising and that any audio effect could be reproduced by a correctly configured generic effect. 
<br/>
**Keywords: audio effects, music information retrieval, differentiable signal processing, sound matching, computer music;**
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

### AutoFx

The implemented architecture for the regression of effects' parameters is dubbed AutoFx and trained according to the following procedure:

<img src="https://adhooge.github.io/AutoFX/figures/network_v0.png" alt="Proposed training framework">

The network architecture being:

<img src="https://adhooge.github.io/AutoFX/figures/autofx.png" alt="Network architecture">

Reconstruction of synthetic sounds is proposed below:


<table>
<caption><b> Modulation</b></caption>
    <tr>
    <td style="text-align: center; vertical-align: middle;"><b>Original</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>Reconstruction</b></td>
</tr>
<tr>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/in_domain/orig_1.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/in_domain/rec_1.wav">
</audio>
</td>
</tr>
<tr>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/in_domain/orig_14.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/in_domain/rec_14.wav">
</audio>
</td>
</tr>
<tr>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/in_domain/orig_19.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/in_domain/rec_19.wav">
</audio>
</td>
</tr>
<tr>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/in_domain/orig_27.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/in_domain/rec_27.wav">
</audio>
</td>
</tr>
</table>

<table>
<caption><b> Delay</b></caption>
    <tr>
    <td style="text-align: center; vertical-align: middle;"><b>Original</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>Reconstruction</b></td>
</tr>
<tr>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/in_domain/orig_30.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/in_domain/rec_30.wav">
</audio>
</td>
</tr>
<tr>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/in_domain/orig_18.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/in_domain/rec_18.wav">
</audio>
</td>
</tr>
<tr>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/in_domain/orig_17.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/in_domain/rec_17.wav">
</audio>
</td>
</tr>
<tr>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/in_domain/orig_16.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/in_domain/rec_16.wav">
</audio>
</td>
</tr>
</table>


<table>
<caption><b> Distortion</b></caption>
    <tr>
    <td style="text-align: center; vertical-align: middle;"><b>Original</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>Reconstruction</b></td>
</tr>
<tr>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/in_domain/orig_3.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/in_domain/rec_3.wav">
</audio>
</td>
</tr>
<tr>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/in_domain/orig_9.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/in_domain/rec_9.wav">
</audio>
</td>
</tr>
<tr>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/in_domain/orig_23.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/in_domain/rec_23.wav">
</audio>
</td>
</tr>
<tr>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/in_domain/orig_31.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/in_domain/rec_31.wav">
</audio>
</td>
</tr>
</table>

#### Reconstruction of Out-of-domain sounds

The following sounds are taken from the online perceptual experiment that was conducted.

<table>
<caption><b> Modulation</b></caption>
    <tr>
    <td style="text-align: center; vertical-align: middle;"><b>Reference</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>163-NC</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>163-C</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>211-C</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>AutoFX</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>AutoFX-F</b></td>
</tr>
<tr>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/ref/modulation_ref_1.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/163-NC/modulation_rec_1.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/163-C/modulation_rec_1.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/211/modulation_rec_1.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/autofx/modulation_rec_1.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/autofx-f/modulation_rec_1.wav">
</audio>
</td>
</tr>
<tr>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/ref/modulation_ref_2.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/163-NC/modulation_rec_2.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/163-C/modulation_rec_2.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/211/modulation_rec_2.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/autofx/modulation_rec_2.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/autofx-f/modulation_rec_2.wav">
</audio>
</td>
</tr>
<tr>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/ref/modulation_ref_3.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/163-NC/modulation_rec_3.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/163-C/modulation_rec_3.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/211/modulation_rec_3.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/autofx/modulation_rec_3.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/autofx-f/modulation_rec_3.wav">
</audio>
</td>
</tr>
</table>

<table>
<caption><b>Delay</b></caption>
    <tr>
    <td style="text-align: center; vertical-align: middle;"><b>Reference</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>163-NC</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>163-C</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>211-C</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>AutoFX</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>AutoFX-F</b></td>
</tr>
<tr>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/ref/delay_ref_1.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/163-NC/delay_rec_1.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/163-C/delay_rec_1.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/211/delay_rec_1.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/autofx/delay_rec_1.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/autofx-f/delay_rec_1.wav">
</audio>
</td>
</tr>
<tr>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/ref/delay_ref_2.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/163-NC/delay_rec_2.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/163-C/delay_rec_2.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/211/delay_rec_2.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/autofx/delay_rec_2.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/autofx-f/delay_rec_2.wav">
</audio>
</td>
</tr>
<tr>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/ref/delay_ref_3.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/163-NC/delay_rec_3.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/163-C/delay_rec_3.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/211/delay_rec_3.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/autofx/delay_rec_3.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/autofx-f/delay_rec_3.wav">
</audio>
</td>
</tr>
</table>

<table>
<caption><b>Distortion</b></caption>
    <tr>
    <td style="text-align: center; vertical-align: middle;"><b>Reference</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>163-NC</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>163-C</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>211-C</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>AutoFX</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>AutoFX-F</b></td>
</tr>
<tr>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/ref/disto_ref_1.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/163-NC/disto_rec_1.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/163-C/disto_rec_1.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/211/disto_rec_1.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/autofx/disto_rec_1.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/autofx-f/disto_rec_1.wav">
</audio>
</td>
</tr>
<tr>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/ref/disto_ref_2.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/163-NC/disto_rec_2.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/163-C/disto_rec_2.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/211/disto_rec_2.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/autofx/disto_rec_2.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/autofx-f/disto_rec_2.wav">
</audio>
</td>
</tr>
<tr>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/ref/disto_ref_3.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/163-NC/disto_rec_3.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/163-C/disto_rec_3.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/211/disto_rec_3.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/autofx/disto_rec_3.wav">
</audio>
</td>
<td style="text-align: center; vertical-align: middle;">
<audio controls>
<source src="https://adhooge.github.io/AutoFX/sounds/autofx/out_domain/autofx-f/disto_rec_3.wav">
</audio>
</td>
</tr>
</table>