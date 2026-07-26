# ruff: noqa: E501
"""Concise movement-only English LTX-2.3 prompts for *Deep Space*."""

from __future__ import annotations

from dataclasses import dataclass


STYLE_ANCHOR = (
    "Photorealistic dark historical fantasy drama, 35mm film grain, wooden smuggler ship hold, amber kerosene lanterns, "
    "wet brown floor, deep brown shadows. Shen Sheng is a lean young Asian man with wet black hair and a soaked patched gray-brown tunic. "
)

SCENE_CONTEXTS = {
    1: "The camera is in the front-right corner beside the starboard bulkhead of one 15-meter by 8-meter lower hold; migrants sit along the walls outside this frame. ",
    2: "The camera is in the front-right corner beside the starboard bulkhead of a 15-meter by 8-meter lower hold; migrants and a second sailor are elsewhere outside frame. ",
    3: "The camera begins beside the starboard bulkhead in the lower hold, then the image hard cuts into Shen Sheng's memories. ",
    4: "The camera is beside the starboard bulkhead of a 15-meter by 8-meter lower hold; wet timber and three lanterns are behind Shen Sheng. ",
    5: "The camera is beside the starboard bulkhead of a 15-meter by 8-meter lower hold and can turn far enough to show Michael, George, and the woman across the room. ",
    6: "The camera faces the middle of a 15-meter by 8-meter lower hold; Shen Sheng is by the starboard bulkhead behind George, while the woman sits nearby with her Bible. ",
    7: "The camera is in the middle of a 15-meter by 8-meter lower hold, moving from George to the woman and the thin man near one timber post. ",
    8: "The camera sees the full width of a 15-meter by 8-meter lower hold: George is near center, the woman is left of him, Shen Sheng is by the starboard wall, and the stairs are at the far end. ",
    9: "The camera is behind Shen Sheng at the starboard wall and faces the far-end stairs of a 15-meter by 8-meter lower hold; George and Michael are between the camera and stairs. ",
    10: "The camera is at the foot of the far-end wooden stairs in a 15-meter by 8-meter lower hold; Shen Sheng, George, the woman, and migrants are below the stairs. ",
    11: "The camera is above the far-end stairs and looks down into a 15-meter by 8-meter lower hold, where the crowd is below the ladder. ",
    12: "The camera is on the floor beside fallen bodies at the foot of the far-end stairs; a crowd is pressed along the walls of a 15-meter by 8-meter lower hold. ",
    13: "The camera is below the ceiling above the far-end stairs and looks into a 15-meter by 8-meter lower hold; Shen Sheng is against the starboard wall below. ",
    14: "The camera is close to Shen Sheng at the starboard bulkhead in a 15-meter by 8-meter lower hold; the ghost is in front of him across the room. ",
    15: "The camera is beside Shen Sheng at the starboard bulkhead in a 15-meter by 8-meter lower hold; the ghost stands between him and the center of the room. ",
}


@dataclass(frozen=True)
class ScenePrompt:
    number: int
    title: str
    duration_seconds: float
    prompt: str

    @property
    def full_prompt(self) -> str:
        return STYLE_ANCHOR + SCENE_CONTEXTS[self.number] + self.prompt


SCENES: tuple[ScenePrompt, ...] = (
    ScenePrompt(1, "Thrown awake", 6.0, "A 35mm camera descends through ceiling beams, sweeps past three swinging lanterns, and settles on Shen Sheng lying in water. A sailor steps in from the right, pulls Shen Sheng up by the collar, and throws him down. Lantern light crosses the sailor's cheek with each swing. Boot splashes, water drops, hull creaks and waves continue. Shen Sheng presses his right hand into the water; the camera ends on expanding ripples."),
    ScenePrompt(2, "Wake and humiliation", 6.0, "From Shen Sheng's right hand at water level, a 40mm camera tracks backward with the ripples and frames him as he pushes up, coughs water, and keeps his hair over his eyes. A sailor crosses the foreground and blocks the lantern on Shen Sheng's left cheek. He looks down and says in English, \"Yellow rat, tell me if you cannot stand. I will kill you quickly.\" He laughs, turns, and walks three splashing steps to the right. After his second step, one low cello note begins beneath the splashes, coughs, drips and wood creaks. Shen Sheng lifts only his pupils toward the sailor's back; the camera ends on his eyes after the final splash."),
    ScenePrompt(3, "Memory cut", 7.0, "An 85mm camera pushes from Shen Sheng's cheek to his closing eyes as lantern light crosses his face and disappears on a heartbeat. Hard cut through a closing orphanage gate, hands striking keyboard keys, a neon city passing in a whip pan, a man running across an ochre street, and a hand coughing blood beside a bed. White flashes and focus shifts end in a dark pupil filling the frame. Drips become keyboard clicks, footsteps, coughs and one heartbeat; a short electronic pulse cuts to black."),
    ScenePrompt(4, "Recognition", 6.0, "From black, a 100mm camera holds Shen Sheng's pupil as a lantern flame moves in its reflection, then pulls back while he opens his eyes, wipes his cheek, lets his mouth corners rise, and closes his lips. The lantern strip moves from his forehead to his chin. A water drop, a hull creak and three fingertip taps enter one by one; a low cello note follows. The camera follows his hand and ends on three more taps."),
    ScenePrompt(5, "Looking around", 7.0, "Starting on Shen Sheng's tapping fingers, a 50mm camera rises and circles clockwise as his eyes move from Michael to the woman with a Bible and then to George. A lantern stripe passes across each face. A blue shape without letters appears once in the puddle and moves across Shen Sheng's knuckles. Finger taps, water, murmurs and cello continue until George draws a breath, stands, and lifts one hand; the camera stops on his hand."),
    ScenePrompt(6, "George speaks", 7.0, "A handheld 50mm camera moves backward as George stands, points into the hold, swallows, looks behind him, and points again. His shadow moves across the wall as a lantern swings. Over water drips and one bowed string note, he says in English, \"There is a ghost. Little Tom died when old John came back. Blue and violet. It smiled. Tom died.\" His hand opens and closes after each phrase. People step back while the woman raises her Bible; the camera follows it upward."),
    ScenePrompt(7, "The Bible and the crowd", 7.0, "The camera follows the raised Bible in a 50mm pan to the woman's face, then to a thin man by a post. Moving lantern bands cross their faces. The woman looks at the stairs and says, \"Father Evan is a priest from Coman Town. The church will protect us.\" The man folds his arms and answers, \"A priest cannot save you from that.\" Three voices behind them say, \"Blue-violet,\" \"It kills,\" and \"The priest will help us.\" Chains, water, voices and a low string note build until George turns his head upward."),
    ScenePrompt(8, "Gunshots", 6.0, "Hard cut to a locked 24mm wide shot. Four gunshots sound above the hold, then two screams; the string note stops. George freezes with his hand raised, the Bible falls, the thin man turns, and Shen Sheng bends his knees and raises a hand. Lantern chains stop moving. One water drop lands in a puddle; after the echo, only its ripple sound remains. The camera ends on the ripple."),
    ScenePrompt(9, "The stairs", 7.0, "A 50mm camera looks over Shen Sheng's shoulder and racks focus from his wet hair to the stairwell. A water drop repeats with no music. George lowers his hand and says, \"Could it be the ghost?\" A blue shape without letters crosses Shen Sheng's cheek and vanishes. Shen Sheng rises from one knee while looking at the stairs; the camera pans left as Michael enters the stairwell light."),
    ScenePrompt(10, "Michael climbs", 7.0, "A low 35mm camera looks up as Michael grabs a ladder rung, looks up, and says, \"No. I am not mad. It is too cramped in here.\" He takes one breath and climbs two rungs while the camera rises below him. His back blocks the deck light and sends a shadow down the ladder. Two migrants climb behind him. Breathing, boots, water and wood creaks continue with one cello note; the camera ends on his boot pressing a plank."),
    ScenePrompt(11, "Fall", 5.0, "Match cut from Michael's boot to a high 28mm view as a tearing sound starts and Michael with two migrants falls straight down. The camera drops briefly and stops over the floor as three impacts land. Michael looks up and exhales white breath. Tearing wood, impacts and crying replace the cello; the camera ends on his breath."),
    ScenePrompt(12, "Cold light", 8.0, "From Michael's white breath at floor level, a 40mm camera slides toward people at the wall as each exhale adds more fog and ice crystals spread over wet boards. Blue-violet light enters through ceiling cracks and covers the amber pools one after another. People cover their mouths and step back. Breath, hum, bow scrape, ice cracks and water continue; a wordless choir begins after the third breath. The camera tilts up with widening violet light."),
    ScenePrompt(13, "Ghost down", 10.0, "From the ceiling crack, a low 28mm camera tilts back as a blue-violet ghost moves through the ceiling. Its lower body changes into smoke, bone spurs move out from its arms, and violet light moves inside its eye sockets. Its light changes faces, water and wood from amber to blue-violet. People step back, kneel, or cover their heads; Shen Sheng presses against the wall and looks up. Hum, ice, choir and breathing continue. The ghost turns its eye sockets toward Shen Sheng until violet light fills the frame."),
    ScenePrompt(14, "Ninety-nine", 7.0, "Match cut from violet light to a 100mm close-up of Shen Sheng's eye, then pull back as violet light covers one cheek and amber light covers the other. Red pulses without letters move in his eye; the choir stops. Over hum and breathing, a synthesized voice says, \"Connection. Ninety-nine percent. Load failed. Reconnecting.\" Shen Sheng moves his eyes to the ghost, swallows, opens and closes his mouth, then turns his head left. The camera pushes in with one alarm beep."),
    ScenePrompt(15, "A ghost", 6.0, "A 35mm over-the-shoulder shot starts behind Shen Sheng as the camera moves toward the ghost. Cold fog crosses Shen Sheng's shoulder, bone spurs move outward, and lantern flames go out one by one. Shen Sheng's back reaches the wall; he exhales once and says, \"There really is a ghost.\" An alarm beep, hum, choir and heartbeat play together, then the choir stops, the alarm stops, and the hum stops. The camera holds his side profile and cuts to black on one heartbeat."),
)


SCENES_BY_NUMBER = {scene.number: scene for scene in SCENES}
