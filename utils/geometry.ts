import { Vector2, HandData } from '../types';

export const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],             // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8],             // Index
  [0, 9], [9, 10], [10, 11], [11, 12],        // Middle
  [0, 13], [13, 14], [14, 15], [15, 16],      // Ring
  [0, 17], [17, 18], [18, 19], [19, 20],      // Pinky
  [5, 9], [9, 13], [13, 17], [0, 5], [0, 17]  // Palm
];

export const getDistance = (p1: Vector2, p2: Vector2): number => {
  return Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2));
};

export const normalizeVector = (v: Vector2): Vector2 => {
  const len = Math.sqrt(v.x * v.x + v.y * v.y);
  if (len === 0) return { x: 0, y: 0 };
  return { x: v.x / len, y: v.y / len };
};

export const scaleVector = (v: Vector2, s: number): Vector2 => {
  return { x: v.x * s, y: v.y * s };
};

export const addVector = (v1: Vector2, v2: Vector2): Vector2 => {
  return { x: v1.x + v2.x, y: v1.y + v2.y };
};

export const subVector = (v1: Vector2, v2: Vector2): Vector2 => {
  return { x: v1.x - v2.x, y: v1.y - v2.y };
};

// Check if hand is a fist based on geometric curl
export const isFist = (hand: HandData): boolean => {
  const wrist = hand.landmarks[0];
  
  // Fingers to check (Index to Pinky)
  // For a sword grip, we mainly care that the fingers are wrapped around the hilt.
  // We compare Tip distance to Wrist vs PIP (knuckle) distance to Wrist.
  const fingers = [
    { tip: 8, pip: 6 },   // Index
    { tip: 12, pip: 10 }, // Middle
    { tip: 16, pip: 14 }, // Ring
    { tip: 20, pip: 18 }  // Pinky
  ];

  let curledCount = 0;
  fingers.forEach(f => {
    const tip = hand.landmarks[f.tip];
    const pip = hand.landmarks[f.pip];
    const mcp = hand.landmarks[f.pip - 1]; // Joint connected to palm
    
    const distTip = getDistance(tip, wrist);
    const distPip = getDistance(pip, wrist);
    const distMcp = getDistance(mcp, wrist);
    
    // Logic: Tip should be closer to wrist than the PIP joint, 
    // OR Tip should be very close to the palm/MCP.
    if (distTip < distPip || distTip < distMcp * 1.2) { 
        curledCount++;
    }
  });

  // Loose check: If 3 or more fingers are curled, it's a fist/grip.
  // We ignore the thumb because in a sword grip, the thumb might be wrapped 
  // or pointing up (saber style).
  return curledCount >= 3;
};

export const getHandCenter = (hand: HandData): Vector2 => {
  const wrist = hand.landmarks[0];
  const indexMCP = hand.landmarks[5];
  const pinkyMCP = hand.landmarks[17];
  
  return {
    x: (wrist.x + indexMCP.x + pinkyMCP.x) / 3,
    y: (wrist.y + indexMCP.y + pinkyMCP.y) / 3
  };
};

export const getSwordDirection = (hand: HandData): Vector2 => {
  const wrist = hand.landmarks[0];
  const indexMCP = hand.landmarks[5];
  const middleMCP = hand.landmarks[9];
  
  // The 'hole' of the fist is roughly between index and middle knuckles
  const topCenter = {
    x: (indexMCP.x + middleMCP.x) / 2,
    y: (indexMCP.y + middleMCP.y) / 2
  };

  const v = { x: topCenter.x - wrist.x, y: topCenter.y - wrist.y };
  return normalizeVector(v);
};

// Returns points to draw the hilt so it looks "inside" the hand
export const getGripPoints = (hand: HandData) => {
    const wrist = hand.landmarks[0];
    const indexMCP = hand.landmarks[5];
    const middleMCP = hand.landmarks[9];

    // Direction vector of the forearm/grip
    const direction = getSwordDirection(hand);

    // Center point between knuckles (Where blade comes out)
    const knuckleCenter = {
        x: (indexMCP.x + middleMCP.x) / 2,
        y: (indexMCP.y + middleMCP.y) / 2
    };

    // Hilt Top (Emitter): Slightly past the knuckles so blade doesn't clip fingers
    const hiltTop = addVector(knuckleCenter, scaleVector(direction, 0.05));

    // Hilt Bottom (Pommel): At the wrist, slightly extended back
    // We use the wrist point directly or slightly shifted back
    const hiltBottom = subVector(wrist, scaleVector(direction, 0.02));

    return { hiltBottom, hiltTop, direction };
};
