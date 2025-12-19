/**
 * World Metadata - OPTIONAL enrichment for pre-generated worlds
 *
 * This file contains OPTIONAL educational content for known worlds.
 * The system works WITHOUT this - it's just enhanced content.
 *
 * For custom/new worlds:
 * - Metadata is loaded from the world's metadata.json file
 * - AI generates contextual narration dynamically
 * - This local cache is NOT required
 *
 * How it works:
 * 1. Check WORLD_METADATA for local enrichment (fast)
 * 2. If not found, fetch from CDN: /worlds/{worldId}/metadata.json
 * 3. If that fails, AI generates content dynamically
 */

// CDN URL for fetching world metadata
const WORLD_CDN_URL = import.meta.env?.VITE_WORLD_CDN_URL || 'http://localhost:8010/worlds';

export interface WorldMetadata {
  id: string;
  name: string;
  subject: string;
  description: string;
  thumbnail?: string;
  positions: Record<string, string>;
  educationalPoints: string[];
  vocabulary: string[];
  suggestedQuestions: string[];
}

/**
 * Get metadata for a world - checks local cache first, then CDN
 */
export async function getWorldMetadata(worldId: string): Promise<WorldMetadata | null> {
  // 1. Check local enrichment cache
  if (WORLD_METADATA[worldId]) {
    return WORLD_METADATA[worldId];
  }

  // 2. Try to fetch from CDN
  try {
    const response = await fetch(`${WORLD_CDN_URL}/${worldId}/metadata.json`);
    if (response.ok) {
      const data = await response.json();
      if (data.world) {
        return {
          id: worldId,
          name: data.world.name || worldId,
          subject: data.world.subject || 'general',
          description: data.world.description || '',
          positions: data.world.positions || {},
          educationalPoints: data.world.educationalPoints || [],
          vocabulary: data.world.vocabulary || [],
          suggestedQuestions: data.world.suggestedQuestions || [],
        };
      }
    }
  } catch {
    // CDN fetch failed, will use dynamic AI
  }

  // 3. Return null - AI will generate dynamically
  return null;
}

/**
 * Create minimal metadata for an unknown world
 */
export function createMinimalMetadata(worldId: string, name?: string): WorldMetadata {
  return {
    id: worldId,
    name: name || worldId.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
    subject: 'exploration',
    description: `Explore ${name || worldId}`,
    positions: {},
    educationalPoints: [],
    vocabulary: [],
    suggestedQuestions: [],
  };
}

// ============================================================================
// OPTIONAL LOCAL ENRICHMENT - Enhanced content for known worlds
// New worlds work WITHOUT being listed here!
// ============================================================================

export const WORLD_METADATA: Record<string, WorldMetadata> = {
  solar_system: {
    id: 'solar_system',
    name: 'Solar System Explorer',
    subject: 'astronomy',
    description: 'Explore our solar system from Mercury to Neptune',
    thumbnail: 'thumbnails/solar_system.jpg',
    positions: {
      start: 'Floating in space, Sun visible in the distance',
      forward: 'Approaching Mercury, the closest planet to the Sun',
      forward_forward: 'Flying past Mercury, Venus coming into view',
      forward_left: 'Turning toward the asteroid belt',
      forward_right: 'Viewing the outer planets in the distance',
      left: 'Looking at the constellation Orion',
      right: 'Mars visible in the distance with its red surface',
      turn_left: 'Rotating to view different star clusters',
      turn_right: 'Turning to see Jupiter and its moons',
      backward: 'Moving back toward the outer solar system',
    },
    educationalPoints: [
      'Mercury is the smallest planet and closest to the Sun',
      'Venus is the hottest planet due to its thick atmosphere trapping heat',
      "Earth is the only planet known to have liquid water on its surface",
      'Mars has the largest volcano in the solar system - Olympus Mons',
      'Jupiter is so large that all other planets could fit inside it',
      "Saturn's rings are made of ice and rock particles",
      'Uranus rotates on its side, unique among planets',
      'Neptune has the strongest winds in the solar system',
    ],
    vocabulary: [
      'orbit',
      'gravity',
      'asteroid',
      'comet',
      'dwarf planet',
      'astronomical unit',
      'light-year',
      'nebula',
    ],
    suggestedQuestions: [
      'Why is Mercury not the hottest planet even though it is closest to the Sun?',
      'What makes Earth special compared to other planets?',
      "How big is Jupiter compared to Earth?",
      "What are Saturn's rings made of?",
    ],
  },

  human_cell: {
    id: 'human_cell',
    name: 'Inside a Human Cell',
    subject: 'biology',
    description: 'Explore the interior of a living cell',
    thumbnail: 'thumbnails/human_cell.jpg',
    positions: {
      start: 'Near the cell membrane, gateway to the cell',
      forward: 'Moving toward the nucleus, the control center',
      forward_forward: 'Approaching the nuclear envelope',
      left: 'Floating toward mitochondria, the powerhouses',
      right: 'Approaching the endoplasmic reticulum',
      turn_left: 'Viewing ribosomes scattered throughout',
      turn_right: 'Looking at the Golgi apparatus',
    },
    educationalPoints: [
      'The cell membrane controls what enters and leaves the cell',
      'The nucleus contains DNA, the genetic blueprint of life',
      'Mitochondria produce ATP, the energy currency of cells',
      'The endoplasmic reticulum helps make proteins and lipids',
      'Ribosomes are the protein factories of the cell',
      'The Golgi apparatus packages and ships proteins',
      'Lysosomes are the recycling centers of the cell',
    ],
    vocabulary: [
      'organelle',
      'cytoplasm',
      'membrane',
      'nucleus',
      'mitochondria',
      'ribosome',
      'ATP',
      'DNA',
    ],
    suggestedQuestions: [
      'Why are mitochondria called the powerhouses of the cell?',
      'What is DNA and where is it found?',
      'How does the cell membrane protect the cell?',
      'Why do cells need energy?',
    ],
  },

  ancient_rome: {
    id: 'ancient_rome',
    name: 'Ancient Roman Forum',
    subject: 'history',
    description: 'Walk through the heart of the Roman Empire',
    thumbnail: 'thumbnails/ancient_rome.jpg',
    positions: {
      start: 'At the entrance to the Roman Forum',
      forward: 'Walking toward the Senate building',
      forward_forward: 'Standing before the Curia Julia',
      left: 'Approaching the Temple of Saturn',
      right: 'Heading toward the Colosseum',
      turn_left: 'Viewing the Arch of Titus',
      turn_right: 'Looking at merchant stalls',
    },
    educationalPoints: [
      'The Roman Forum was the center of public life in ancient Rome',
      'The Senate made laws and governed the Republic',
      'Roman architecture influenced buildings for centuries',
      'Latin, the Roman language, is the basis for many modern languages',
      'Romans built roads, aqueducts, and innovative structures',
      'Gladiators fought in the Colosseum for entertainment',
    ],
    vocabulary: [
      'senate',
      'republic',
      'emperor',
      'gladiator',
      'aqueduct',
      'forum',
      'toga',
      'legion',
    ],
    suggestedQuestions: [
      'How was the Roman Republic governed?',
      'What happened in the Colosseum?',
      'How did Romans get fresh water to their city?',
      'What language did Romans speak?',
    ],
  },

  ocean_deep: {
    id: 'ocean_deep',
    name: 'Deep Ocean Explorer',
    subject: 'marine_biology',
    description: 'Dive into the mysterious deep ocean',
    thumbnail: 'thumbnails/ocean_deep.jpg',
    positions: {
      start: 'Descending through the twilight zone',
      forward: 'Going deeper toward the abyssal plain',
      forward_forward: 'Reaching the midnight zone',
      left: 'Approaching a hydrothermal vent',
      right: 'Swimming toward a whale fall ecosystem',
      turn_left: 'Viewing bioluminescent creatures',
      turn_right: 'Observing giant isopods',
    },
    educationalPoints: [
      'The ocean has distinct zones based on depth and light',
      'Bioluminescence helps creatures communicate and hunt in darkness',
      'Hydrothermal vents support unique ecosystems without sunlight',
      'Pressure increases dramatically with depth',
      'Many deep-sea creatures have never been studied',
      'Whale falls create temporary ecosystems on the ocean floor',
    ],
    vocabulary: [
      'bioluminescence',
      'abyssal',
      'pressure',
      'hydrothermal',
      'benthic',
      'plankton',
      'adaptation',
    ],
    suggestedQuestions: [
      'Why do deep-sea creatures glow?',
      'How do animals survive at extreme depths?',
      'What lives near hydrothermal vents?',
      'Why is the deep ocean so dark?',
    ],
  },

  volcano_interior: {
    id: 'volcano_interior',
    name: 'Inside a Volcano',
    subject: 'geology',
    description: 'Explore the interior of an active volcano',
    thumbnail: 'thumbnails/volcano_interior.jpg',
    positions: {
      start: 'Near the volcano rim looking down',
      forward: 'Descending into the crater',
      forward_forward: 'Approaching the magma chamber',
      left: 'Viewing ancient lava tubes',
      right: 'Observing the volcanic vent',
      turn_left: 'Seeing layers of volcanic rock',
      turn_right: 'Looking at sulfur deposits',
    },
    educationalPoints: [
      'Volcanoes form at tectonic plate boundaries',
      'Magma is molten rock beneath the surface',
      'Lava is magma that reaches the surface',
      'Volcanic eruptions can be explosive or effusive',
      'Volcanoes create new land and enrich soil',
      'The Ring of Fire has most of Earths volcanoes',
    ],
    vocabulary: [
      'magma',
      'lava',
      'tectonic',
      'eruption',
      'crater',
      'igneous',
      'seismic',
      'mantle',
    ],
    suggestedQuestions: [
      'What is the difference between magma and lava?',
      'Why do volcanoes erupt?',
      'What is the Ring of Fire?',
      'Can volcanoes be beneficial?',
    ],
  },

  rainforest_canopy: {
    id: 'rainforest_canopy',
    name: 'Amazon Rainforest Canopy',
    subject: 'ecology',
    description: 'Explore the worlds most biodiverse ecosystem',
    thumbnail: 'thumbnails/rainforest_canopy.jpg',
    positions: {
      start: 'Emerging through the forest floor',
      forward: 'Rising up through the understory',
      forward_forward: 'Breaking through to the canopy',
      left: 'Following a river through the forest',
      right: 'Approaching a massive ceiba tree',
      turn_left: 'Spotting colorful macaws',
      turn_right: 'Observing a sloth in the trees',
    },
    educationalPoints: [
      'Rainforests contain over half of all species on Earth',
      'The canopy receives the most sunlight',
      'Many rainforest plants are used in medicine',
      'Rainforests produce about 20% of Earths oxygen',
      'Deforestation threatens biodiversity',
      'Indigenous peoples have lived in rainforests for thousands of years',
    ],
    vocabulary: [
      'biodiversity',
      'canopy',
      'understory',
      'ecosystem',
      'deforestation',
      'species',
      'habitat',
    ],
    suggestedQuestions: [
      'Why are rainforests called the lungs of the Earth?',
      'What animals live in the canopy layer?',
      'Why is biodiversity important?',
      'How can we protect rainforests?',
    ],
  },

  dna_helix: {
    id: 'dna_helix',
    name: 'DNA Double Helix',
    subject: 'molecular_biology',
    description: 'Journey along the blueprint of life',
    thumbnail: 'thumbnails/dna_helix.jpg',
    positions: {
      start: 'Outside the DNA molecule',
      forward: 'Traveling along the major groove',
      forward_forward: 'Deep inside the helix structure',
      left: 'Viewing the minor groove',
      right: 'Approaching a replication fork',
      turn_left: 'Seeing base pairs up close',
      turn_right: 'Observing protein enzymes at work',
    },
    educationalPoints: [
      'DNA carries genetic information in all living things',
      'The double helix was discovered by Watson and Crick',
      'Four bases (A, T, G, C) encode genetic information',
      'A pairs with T, G pairs with C',
      'DNA replication allows cells to copy genetic information',
      'Mutations are changes in the DNA sequence',
    ],
    vocabulary: [
      'nucleotide',
      'base pair',
      'helix',
      'replication',
      'gene',
      'chromosome',
      'mutation',
      'enzyme',
    ],
    suggestedQuestions: [
      'What do the letters A, T, G, and C represent?',
      'How does DNA copy itself?',
      'What are genes?',
      'How is DNA related to traits like eye color?',
    ],
  },

  medieval_castle: {
    id: 'medieval_castle',
    name: 'Medieval Castle',
    subject: 'medieval_history',
    description: 'Explore a medieval fortress',
    thumbnail: 'thumbnails/medieval_castle.jpg',
    positions: {
      start: 'Crossing the drawbridge',
      forward: 'Entering the main courtyard',
      forward_forward: 'Approaching the keep',
      left: 'Walking toward the armory',
      right: 'Heading to the great hall',
      turn_left: 'Viewing the castle walls',
      turn_right: 'Looking at the stables',
    },
    educationalPoints: [
      'Castles were built for defense and to show power',
      'The keep was the strongest part of the castle',
      'Knights trained from childhood to fight',
      'Feudalism organized medieval society',
      'Siege warfare tried to break castle defenses',
      'Life in a castle was difficult for most people',
    ],
    vocabulary: [
      'knight',
      'feudalism',
      'keep',
      'moat',
      'drawbridge',
      'siege',
      'lord',
      'serf',
    ],
    suggestedQuestions: [
      'How did castles protect against attacks?',
      'What was life like for a medieval knight?',
      'What is feudalism?',
      'How were castles built without modern tools?',
    ],
  },

  atom_structure: {
    id: 'atom_structure',
    name: 'Inside an Atom',
    subject: 'physics',
    description: 'Explore the quantum realm of atoms',
    thumbnail: 'thumbnails/atom_structure.jpg',
    positions: {
      start: 'Outside the electron cloud',
      forward: 'Passing through the electron shells',
      forward_forward: 'Approaching the nucleus',
      left: 'Viewing the s orbital',
      right: 'Observing the p orbital shapes',
      turn_left: 'Seeing electrons as probability clouds',
      turn_right: 'Looking at the nucleus up close',
    },
    educationalPoints: [
      'Atoms are the building blocks of matter',
      'The nucleus contains protons and neutrons',
      'Electrons orbit in probability clouds called orbitals',
      'Protons are positive, electrons are negative',
      'Most of an atom is empty space',
      'Quantum mechanics describes atomic behavior',
    ],
    vocabulary: [
      'electron',
      'proton',
      'neutron',
      'orbital',
      'nucleus',
      'quantum',
      'energy level',
      'charge',
    ],
    suggestedQuestions: [
      'What are atoms made of?',
      'Why dont electrons fall into the nucleus?',
      'What makes different elements different?',
      'How small is an atom?',
    ],
  },

  egyptian_pyramid: {
    id: 'egyptian_pyramid',
    name: 'Egyptian Pyramid Interior',
    subject: 'ancient_history',
    description: 'Explore the mysteries of the Great Pyramid',
    thumbnail: 'thumbnails/egyptian_pyramid.jpg',
    positions: {
      start: 'At the pyramid entrance',
      forward: 'Walking up the grand gallery',
      forward_forward: 'Approaching the Kings Chamber',
      left: 'Entering the Queens Chamber passage',
      right: 'Discovering a hidden passage',
      turn_left: 'Viewing hieroglyphics on walls',
      turn_right: 'Looking at ancient artifacts',
    },
    educationalPoints: [
      'The Great Pyramid was built around 2560 BCE',
      'Pyramids were tombs for pharaohs',
      'Hieroglyphics were the Egyptian writing system',
      'Ancient Egyptians believed in an afterlife',
      'Building the pyramids required incredible engineering',
      'Many chambers remain unexplored to this day',
    ],
    vocabulary: [
      'pharaoh',
      'hieroglyphics',
      'sarcophagus',
      'mummy',
      'tomb',
      'artifact',
      'archaeology',
    ],
    suggestedQuestions: [
      'How were the pyramids built?',
      'Why did Egyptians build pyramids?',
      'What is mummification?',
      'How do we read hieroglyphics today?',
    ],
  },
};

/**
 * Get all available worlds
 */
export function getAvailableWorlds(): WorldMetadata[] {
  return Object.values(WORLD_METADATA);
}

/**
 * Get worlds by subject
 */
export function getWorldsBySubject(subject: string): WorldMetadata[] {
  return Object.values(WORLD_METADATA).filter((w) => w.subject === subject);
}

/**
 * Get a world by ID
 */
export function getWorldById(id: string): WorldMetadata | undefined {
  return WORLD_METADATA[id];
}

export default WORLD_METADATA;
