export interface Vector2 {
  x: number;
  y: number;
}

export enum GameMode {
  SOLO = 'SOLO',
  DUEL = 'DUEL'
}

export enum GameState {
  IDLE = 'IDLE',           // Waiting for user
  HILT_FORMING = 'HILT',   // Left fist detected, hilt particles appearing
  CASTING = 'CASTING',     // Right hand pulling blade out
  ACTIVE = 'ACTIVE',       // Fully formed sword
  DISSOLVING = 'DISSOLVING', // Sword turning off
  
  // Duel specific states
  DUEL_WAITING = 'DUEL_WAITING', // Waiting for 2 players
  DUEL_BOWING = 'DUEL_BOWING',   // Waiting for bow
  DUEL_FIGHTING = 'DUEL_FIGHTING', // Combat active
  DUEL_ENDED = 'DUEL_ENDED'      // Match over
}

export interface Particle {
  id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
  maxLife: number;
  size: number;
  color: string;
  type: 'core' | 'glow' | 'spark' | 'trail' | 'shockwave';
}

export interface HandData {
  landmarks: { x: number; y: number; z: number }[];
  handedness: 'Left' | 'Right';
  worldLandmarks?: { x: number; y: number; z: number }[]; // Optional 3D coords
}

export interface DuelPlayerState {
  id: 1 | 2;
  hp: number;
  maxHp: number;
  saberColor: string;
  swordState: GameState;
  swordProgress: number;
  swordHand: 'Left' | 'Right' | null;
  lastFistTime: number;
  velocity: number;
  prevHandPos: Vector2 | null;
  trail: { tip: Vector2; base: Vector2; color: string }[];
  lastHitTime: number; // Invulnerability frames
}