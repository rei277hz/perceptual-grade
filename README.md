# Perceptual Grade

This repository is a downstream code snapshot of **Perceptual Grade v1.23.2**,
built on **2026-11-01** and extracted from
[`perceptual_grade-v1.23.2-20261101.zip`](https://rei277hz.github.io/perceptual-grade/downloads/perceptual_grade-v1.23.2-20261101.zip).
It currently contains the preserved Perceptual Grade implementation, **not
Perceptual Grade Next**.

Development sources, build scripts, and tests are maintained in
[rei277hz/chromaticity](https://github.com/rei277hz/chromaticity), under
[`products/perceptual-grade-blender`](https://github.com/rei277hz/chromaticity/tree/main/products/perceptual-grade-blender)
and its shared libraries. This snapshot repository accepts
[issues](https://github.com/rei277hz/perceptual-grade/issues), but does not accept
pull requests. Submit code, documentation, and translation changes as
[pull requests to chromaticity](https://github.com/rei277hz/chromaticity/pulls).

The [Perceptual Grade Handbook](https://rei277hz.github.io/perceptual-grade/)
documents the legacy 1.23.2 workflow and hosts this snapshot's download.
**The handbook has not been updated for this development build or for Perceptual
Grade Next.** Use it for background and the grading model; follow this README
and the [current product documentation](https://github.com/rei277hz/chromaticity/blob/main/products/perceptual-grade-blender/README.md)
for the packaged implementation's controls and limitations.

Perceptual Grade provides color grading for Blender's Shader Editor and
Compositor, with reusable Grades, ordered Grade Stacks, image baking, and
interactive diagrams and color scopes.

## Install

1. Use Blender **5.2 or later**. Recorded Linux acceptance checks use Blender 5.2.2.
2. Download the snapshot ZIP linked above. In Blender, choose **Install from
   Disk**, select the ZIP, and enable **Perceptual Grade**. Install the complete
   archive and keep its component packages together.
3. Open a Shader Editor or Compositor with an editable node tree.

When upgrading from an installation with the old `_vendor` package layout,
disable it before replacement and restart Blender before enabling this version.
Perceptual Grade and Next share the extension identity `perceptual_grade`;
compare them in separate Blender profiles or processes.

## Add a Grade or Grade Stack

In the node editor's **Add → Color** menu, choose **Perceptual Grade** or
**Perceptual Grade Stack**. These actions are also available through **F3**
search. Connect the new node into the color-processing path, select it, and
open the sidebar with **N**. Choose the **Perceptual Grade** tab.

A Grade exposes three stages: **1 · Linear**, **2 · Tone**, and **3 · Color**.
The Grade selector initially has no selection; the first grading edit creates
a reusable Grade. Selecting the same Grade on other nodes shares its settings.
Use **New Grade** to create an independent copy of a selected Grade.

For a Grade Stack, use **Add Grade**, select a Grade for each entry, and arrange
the entries in order. Entries can be enabled or disabled and evaluated in the
Forward or Inverse direction. The pencil action, **Edit Grade**, edits the
shared Grade used by that entry.

## Bake an Image

On a Grade or Grade Stack node, choose a managed packed Image with its **Image**
selector, or leave the selector empty to create and bind one when baking
succeeds. Click **Bake Grade Output**. Progress and errors appear on that node.

Shader nodes determine and display the bake route. A supported still Image
Texture with an unconnected Vector input uses an **Image bake** at the source
image's dimensions. Connected source coordinates or surface-dependent inputs
select **UV bake · Cycles**, which requires Cycles, an active mesh, an active
render UV map, and the appropriate material.

Baking produces a separate **Baked Color** output, plus **Baked Alpha** on
Shader nodes. Existing Color connections continue to use live grading; connect
the baked outputs explicitly to use the stored result. The **Perceptual Grade
Bake** node in **Add → Color** reads an existing baked Image.

Rebaking a shared destination Image updates its consumers. Starting timeline
playback cancels unfinished baking and preserves previously published results.

## Diagrams and scopes

Select a Grade and click **Show Diagram** in the **Perceptual Grade** sidebar.
For a Stack entry, use **Edit Grade** to open its Grade controls. Choose the
Linear, Tone, or Color stage to work with its diagram.

Diagrams and scopes appear in the **node editor**. The stages provide
vectorscopes, waveforms, and histograms for inspecting the image and grading
controls. Sampling pauses during timeline playback.

## Development limits

- This is a development snapshot. Saved-data and kernel layouts can change;
  automatic migration from older schemas is not provided.
- Bake operates on Grade and Grade Stack nodes. Arbitrary-node baking and
  automatic rewiring are not supported.
- Shader Bake excludes UDIM, movie/image-sequence, multilayer/multiview sources,
  and unsupported UV layouts. See the product documentation for the full
  eligibility rules.
- Linux/Blender checks do not establish macOS/Metal or physical HDR-display
  behavior; those acceptance areas remain incomplete.

See the [preserved product's documentation](https://github.com/rei277hz/chromaticity/blob/main/products/perceptual-grade-blender/README.md)
for detailed behavior and validation. The separate
[Perceptual Grade Next documentation](https://github.com/rei277hz/chromaticity/blob/main/products/perceptual-grade-next-blender/README.md)
describes that implementation's status.

## Repository contents

The packaged components sit directly beneath the repository root:

| Entry | Purpose |
| --- | --- |
| `README.md` | Snapshot provenance, installation, and usage guide |
| `blender_manifest.toml` | Extension identity, version, and Blender requirements |
| `__init__.py` | Extension registration entrypoint |
| `perceptual_grade_blender/` | Blender controls, node graphs, scopes, and baking |
| `perceptual_grade_core/` | Grading models, Grade/Stack state, localization, and policies |
| `blender_runtime/` | Shared Blender integration and helper runtime |
| `color_core/` | Shared color mathematics |
| `invariant/` | Runtime contracts and state-machine support |
| `LICENSE` | GNU General Public License, version 3 |

The extension manifest declares **GPL-3.0-or-later**. See [LICENSE](LICENSE).

[Development source](https://github.com/rei277hz/chromaticity) ·
[Report an issue](https://github.com/rei277hz/perceptual-grade/issues) ·
[Submit a pull request upstream](https://github.com/rei277hz/chromaticity/pulls)
