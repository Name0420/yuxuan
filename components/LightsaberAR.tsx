import React, { useEffect, useRef, useState, useCallback } from 'react';
import Webcam from 'react-webcam';
import { FilesetResolver, HandLandmarker, FaceLandmarker } from '@mediapipe/tasks-vision';
import { GameState, GameMode, Particle, HandData, Vector2, DuelPlayerState } from '../types';
import { 
  getDistance, 
  isFist, 
  getHandCenter, 
  scaleVector, 
  addVector, 
  getGripPoints,
  HAND_CONNECTIONS
} from '../utils/geometry';

const HAND_MODEL_PATH = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const FACE_MODEL_PATH = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

// Enemies text content
const BAD_THINGS = ["早八", "工作", "上学", "作业", "考试", "DDL", "脱发", "内卷", "加班", "周一", "Bug"];

interface Enemy {
  id: number;
  x: number;
  y: number;
  text: string;
  speed: number;
  hit: boolean;
}

// --- GEOMETRY HELPERS ---

// Distance from point p to segment [v, w]
const distToSegment = (p: Vector2, v: Vector2, w: Vector2) => {
  const l2 = (w.x - v.x)**2 + (w.y - v.y)**2;
  if (l2 === 0) return Math.sqrt((p.x - v.x)**2 + (p.y - v.y)**2); 
  let t = ((p.x - v.x) * (w.x - v.x) + (p.y - v.y) * (w.y - v.y)) / l2;
  t = Math.max(0, Math.min(1, t));
  const px = v.x + t * (w.x - v.x);
  const py = v.y + t * (w.y - v.y);
  return Math.sqrt((p.x - px)**2 + (p.y - py)**2);
};

const lineRectCollide = (
  x1: number, y1: number, x2: number, y2: number,
  rx: number, ry: number, rw: number, rh: number
) => {
  const inside = (x: number, y: number) => x >= rx && x <= rx + rw && y >= ry && y <= ry + rh;
  if (inside(x1, y1) || inside(x2, y2)) return true;

  const lineLine = (x1: number, y1: number, x2: number, y2: number, x3: number, y3: number, x4: number, y4: number) => {
    const denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1);
    if (denom === 0) return false;
    const ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom;
    const ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom;
    return ua >= 0 && ua <= 1 && ub >= 0 && ub <= 1;
  };

  const right = rx + rw;
  const bottom = ry + rh;
  
  if (lineLine(x1, y1, x2, y2, rx, ry, right, ry)) return true;
  if (lineLine(x1, y1, x2, y2, rx, bottom, right, bottom)) return true;
  if (lineLine(x1, y1, x2, y2, rx, ry, rx, bottom)) return true;
  if (lineLine(x1, y1, x2, y2, right, ry, right, bottom)) return true;

  return false;
};

const getLineIntersection = (p0: Vector2, p1: Vector2, p2: Vector2, p3: Vector2): Vector2 | null => {
    const s1_x = p1.x - p0.x;
    const s1_y = p1.y - p0.y;
    const s2_x = p3.x - p2.x;
    const s2_y = p3.y - p2.y;

    const denom = -s2_x * s1_y + s1_x * s2_y;
    if (denom === 0) return null;

    const s = (-s1_y * (p0.x - p2.x) + s1_x * (p0.y - p2.y)) / denom;
    const t = (s2_x * (p0.y - p2.y) - s2_y * (p0.x - p2.x)) / denom;

    if (s >= 0 && s <= 1 && t >= 0 && t <= 1) {
        return {
            x: p0.x + (t * s1_x),
            y: p0.y + (t * s1_y)
        };
    }
    return null;
};

const LightsaberAR: React.FC = () => {
  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [loading, setLoading] = useState(true);
  
  // Mode Selection
  const [gameMode, setGameMode] = useState<GameMode>(GameMode.SOLO);
  
  // Solo State
  const [gameState, setGameState] = useState<GameState>(GameState.IDLE);
  const [isGameActive, setIsGameActive] = useState(false);
  const [score, setScore] = useState(0);

  // Duel State
  const [duelWinner, setDuelWinner] = useState<number | null>(null);
  const [p1Hp, setP1Hp] = useState(100);
  const [p2Hp, setP2Hp] = useState(100);
  const [hitFlash, setHitFlash] = useState<number>(0); // 0=none, 1=p1 hit, 2=p2 hit
  const [clashFlash, setClashFlash] = useState<boolean>(false);

  // --- Refs ---
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const requestRef = useRef<number | null>(null);
  const particlesRef = useRef<Particle[]>([]);
  
  // Solo Logic Refs
  const gameStateRef = useRef<GameState>(GameState.IDLE);
  const swordProgressRef = useRef<number>(0); 
  const prevHandPosRef = useRef<Vector2 | null>(null);
  const velocityRef = useRef<number>(0); 
  const trailHistoryRef = useRef<{ tip: Vector2, base: Vector2, color: string }[]>([]);
  const enemiesRef = useRef<Enemy[]>([]);
  const lastEnemySpawnTime = useRef<number>(0);
  const noseHistoryRef = useRef<{y: number, time: number}[]>([]);
  const lastNodTriggerTime = useRef<number>(0);
  const isGameActiveRef = useRef<boolean>(false);
  const swordHandRef = useRef<'Left' | 'Right' | null>(null);
  const lastFistTimeRef = useRef<number>(0);

  // Duel Logic Refs
  const duelStateRef = useRef<GameState>(GameState.DUEL_WAITING);
  const player1Ref = useRef<DuelPlayerState>({
    id: 1, hp: 100, maxHp: 100, saberColor: '#00ffff', // Cyan
    swordState: GameState.IDLE, swordProgress: 0, swordHand: null,
    lastFistTime: 0, velocity: 0, prevHandPos: null, trail: [], lastHitTime: 0
  });
  const player2Ref = useRef<DuelPlayerState>({
    id: 2, hp: 100, maxHp: 100, saberColor: '#ff3333', // Red
    swordState: GameState.IDLE, swordProgress: 0, swordHand: null,
    lastFistTime: 0, velocity: 0, prevHandPos: null, trail: [], lastHitTime: 0
  });

  // Constants
  const SWORD_LENGTH_SOLO = 0.62;
  const SWORD_LENGTH_DUEL = 0.38; 
  const DUEL_AUTO_IGNITE_TIME = 400; // Faster ignition (0.4s)
  const CASTING_START_DIST = 0.15; 
  const CASTING_COMPLETE_DIST = 0.55;
  const FIST_LOSS_GRACE_PERIOD = 2500; // HUGE grace period (2.5s) to prevent sword loss during combat

  const setupMediaPipe = async () => {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );
    
    handLandmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: HAND_MODEL_PATH, delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: 4, 
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    faceLandmarkerRef.current = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: FACE_MODEL_PATH, delegate: "GPU" },
      runningMode: "VIDEO",
      numFaces: 2, 
      minFaceDetectionConfidence: 0.5,
      minFacePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    setLoading(false);
  };

  useEffect(() => {
    setupMediaPipe();
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
      handLandmarkerRef.current?.close();
      faceLandmarkerRef.current?.close();
    };
  }, []);

  const resetGame = () => {
      setScore(0);
      setIsGameActive(false);
      isGameActiveRef.current = false;
      setGameState(GameState.IDLE);
      gameStateRef.current = GameState.IDLE;
      enemiesRef.current = [];
      setHitFlash(0);
      setClashFlash(false);
      
      duelStateRef.current = GameState.DUEL_WAITING;
      setDuelWinner(null);
      setP1Hp(100);
      setP2Hp(100);
      
      const resetPlayer = (p: DuelPlayerState) => {
          p.hp = 100;
          p.swordState = GameState.IDLE;
          p.swordProgress = 0;
          p.swordHand = null;
          p.trail = [];
          p.lastHitTime = 0;
          p.lastFistTime = 0;
      };
      resetPlayer(player1Ref.current);
      resetPlayer(player2Ref.current);
  };

  const toggleMode = () => {
      const newMode = gameMode === GameMode.SOLO ? GameMode.DUEL : GameMode.SOLO;
      setGameMode(newMode);
      resetGame();
  };

  const spawnParticles = (
    x: number, y: number, count: number, type: Particle['type'], 
    colorOverride?: string, velocityScale: number = 0.01, 
    sizeBase: number = 3, lifeMaxOverride?: number
  ) => {
    for (let i = 0; i < count; i++) {
      const angle = Math.random() * Math.PI * 2;
      const speed = Math.random() * velocityScale;
      particlesRef.current.push({
        id: Math.random(),
        x: x, y: y,
        vx: Math.cos(angle) * speed, vy: Math.sin(angle) * speed,
        life: 1.0,
        maxLife: lifeMaxOverride || (0.3 + Math.random() * 0.4),
        size: Math.random() * sizeBase + 0.5,
        color: colorOverride || '#fff',
        type
      });
    }
  };

  const detectSoloNod = (noseTip: {x: number, y: number}) => {
      const now = Date.now();
      if (now - lastNodTriggerTime.current < 1500) return;
      const history = noseHistoryRef.current;
      history.push({ y: noseTip.y, time: now });
      while(history.length > 0 && now - history[0].time > 500) history.shift();
      if (history.length < 5) return;
      const startY = history[0].y;
      let maxY = startY;
      for(const point of history) if (point.y > maxY) maxY = point.y;
      const range = maxY - startY;
      if (range > 0.04) {
          const lastY = history[history.length-1].y;
          if (maxY - lastY > 0.02) {
              lastNodTriggerTime.current = now;
              isGameActiveRef.current = !isGameActiveRef.current;
              setIsGameActive(isGameActiveRef.current);
              if (!isGameActiveRef.current) enemiesRef.current = [];
              else setScore(0);
              spawnParticles(0.5, 0.5, 20, 'shockwave', isGameActiveRef.current ? '#00ff00' : '#ff0000', 0.02, 5);
          }
      }
  };

  const processDuelPlayer = (
      player: DuelPlayerState, 
      myHands: HandData[]
    ) => {
    const now = Date.now();
    let swordHand = myHands.find(h => h.handedness === player.swordHand);

    switch (player.swordState) {
        case GameState.IDLE:
             const fistHand = myHands.find(h => isFist(h));
             if (fistHand) {
                 player.swordHand = fistHand.handedness;
                 player.swordState = GameState.HILT_FORMING; 
                 player.lastFistTime = now;
                 player.swordProgress = 0;
             }
             break;

        case GameState.HILT_FORMING: 
            if (swordHand && isFist(swordHand)) {
                player.swordProgress += 0.05; 
                if (now - player.lastFistTime > DUEL_AUTO_IGNITE_TIME) {
                    player.swordState = GameState.ACTIVE;
                    player.swordProgress = 1;
                    const { hiltTop } = getGripPoints(swordHand);
                    spawnParticles(hiltTop.x, hiltTop.y, 20, 'shockwave', player.saberColor, 0.05, 4);
                } else {
                    const { hiltTop } = getGripPoints(swordHand);
                    spawnParticles(hiltTop.x, hiltTop.y, 2, 'glow', player.saberColor, 0.01, 1);
                }
            } else {
                player.swordState = GameState.IDLE;
                player.swordHand = null;
                player.swordProgress = 0;
            }
            break;

        case GameState.ACTIVE:
            if (swordHand && isFist(swordHand)) {
                player.lastFistTime = now;
            }
            // Strict check removed, long grace period
            if (now - player.lastFistTime > FIST_LOSS_GRACE_PERIOD) {
                player.swordState = GameState.DISSOLVING;
            }
            if (swordHand) {
                const { hiltTop } = getGripPoints(swordHand);
                if (player.prevHandPos) {
                    const delta = getDistance(hiltTop, player.prevHandPos);
                    player.velocity = player.velocity * 0.85 + delta * 0.15;
                }
                player.prevHandPos = hiltTop;
            }
            break;

        case GameState.DISSOLVING:
            player.swordProgress -= 0.1; 
            if (player.swordProgress <= 0) {
                player.swordProgress = 0;
                player.swordState = GameState.IDLE;
                player.swordHand = null;
            }
            break;
    }
  };

  const processSoloPlayer = (hands: HandData[]) => {
    const now = Date.now();
    let swordHand = hands.find(h => h.handedness === swordHandRef.current);
    let forceHand = hands.find(h => h.handedness !== swordHandRef.current);

    switch (gameStateRef.current) {
      case GameState.IDLE:
        const fistHand = hands.find(h => isFist(h));
        if (fistHand) {
          swordHandRef.current = fistHand.handedness;
          gameStateRef.current = GameState.HILT_FORMING;
          setGameState(GameState.HILT_FORMING);
          lastFistTimeRef.current = now;
        }
        break;
      case GameState.HILT_FORMING:
      case GameState.CASTING:
      case GameState.ACTIVE:
        if (swordHand && isFist(swordHand)) lastFistTimeRef.current = now;
        if (now - lastFistTimeRef.current > 500) { // Solo mode grace period (shorter)
           gameStateRef.current = GameState.DISSOLVING;
           setGameState(GameState.DISSOLVING);
        }
        if (gameStateRef.current === GameState.HILT_FORMING && swordHand && forceHand) {
           if (getDistance(getHandCenter(swordHand), getHandCenter(forceHand)) < CASTING_START_DIST) {
              gameStateRef.current = GameState.CASTING;
              setGameState(GameState.CASTING);
           }
        }
        if (gameStateRef.current === GameState.CASTING && swordHand && forceHand) {
           const dist = getDistance(getHandCenter(swordHand), getHandCenter(forceHand));
           const progress = Math.min((dist - CASTING_START_DIST) / (CASTING_COMPLETE_DIST - CASTING_START_DIST), 1);
           swordProgressRef.current = Math.max(0, progress);
           spawnParticles(getHandCenter(forceHand).x, getHandCenter(forceHand).y, 2, 'spark', '#00ccff', 0.02, 2.5);
           if (progress >= 1) {
             gameStateRef.current = GameState.ACTIVE;
             setGameState(GameState.ACTIVE);
             const { hiltTop } = getGripPoints(swordHand);
             spawnParticles(hiltTop.x, hiltTop.y, 50, 'shockwave', '#ffffff', 0.05, 4);
           }
        }
        if (gameStateRef.current === GameState.ACTIVE && swordHand) {
           const { hiltTop } = getGripPoints(swordHand);
           if (prevHandPosRef.current) {
             const delta = getDistance(hiltTop, prevHandPosRef.current);
             velocityRef.current = velocityRef.current * 0.85 + delta * 0.15;
           }
           prevHandPosRef.current = hiltTop;
        }
        break;
      case GameState.DISSOLVING:
        swordProgressRef.current -= 0.05;
        if (swordProgressRef.current <= 0) {
          swordProgressRef.current = 0;
          gameStateRef.current = GameState.IDLE;
          setGameState(GameState.IDLE);
          swordHandRef.current = null;
        }
        break;
    }
  };

  const updateSoloGameLogic = (canvasWidth: number, canvasHeight: number, swordSegment: {p1: Vector2, p2: Vector2} | null) => {
      if (!isGameActiveRef.current) return;
      const now = Date.now();

      if (now - lastEnemySpawnTime.current > 1200) { 
          const text = BAD_THINGS[Math.floor(Math.random() * BAD_THINGS.length)];
          enemiesRef.current.push({
              id: Math.random(),
              x: 0.1 + Math.random() * 0.8,
              y: -0.1,
              text: text,
              speed: 0.003 + Math.random() * 0.004,
              hit: false
          });
          lastEnemySpawnTime.current = now;
      }

      for (let i = enemiesRef.current.length - 1; i >= 0; i--) {
          const enemy = enemiesRef.current[i];
          enemy.y += enemy.speed;
          let isHit = false;

          if (swordSegment && gameStateRef.current === GameState.ACTIVE) {
              const s1 = { x: swordSegment.p1.x * canvasWidth, y: swordSegment.p1.y * canvasHeight };
              const s2 = { x: swordSegment.p2.x * canvasWidth, y: swordSegment.p2.y * canvasHeight };
              const ex = (1 - enemy.x) * canvasWidth;
              const ey = enemy.y * canvasHeight;
              const totalWidth = enemy.text.length * 35 + 30; 
              const totalHeight = 70; 
              const rx = ex - totalWidth / 2;
              const ry = ey - totalHeight / 2;
              if (lineRectCollide(s1.x, s1.y, s2.x, s2.y, rx, ry, totalWidth, totalHeight)) {
                  isHit = true;
              }
          }

          if (isHit) {
              enemiesRef.current.splice(i, 1);
              setScore(s => s + 1);
              spawnParticles(enemy.x, enemy.y, 15, 'spark', 'rgba(255, 100, 100, 0.5)', 0.015, 1.5, 0.4);
          } else if (enemy.y > 1.1) {
              enemiesRef.current.splice(i, 1);
          }
      }
  };

  const drawSkeleton = (ctx: CanvasRenderingContext2D, hand: HandData, width: number, height: number, color: string) => {
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
      const p1 = hand.landmarks[startIdx];
      const p2 = hand.landmarks[endIdx];
      const x1 = (1 - p1.x) * width;
      const y1 = p1.y * height;
      const x2 = (1 - p2.x) * width;
      const y2 = p2.y * height;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }
    for (const p of hand.landmarks) {
      const x = (1 - p.x) * width;
      const y = p.y * height;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  };

  const drawSaber = (
      ctx: CanvasRenderingContext2D, 
      hand: HandData, 
      swordState: GameState, 
      progress: number,
      velocity: number,
      trail: any[],
      color: string,
      width: number, 
      height: number,
      maxLength: number
  ) => {
      if (swordState === GameState.IDLE) return null;

      const { hiltBottom, hiltTop, direction } = getGripPoints(hand);
      const bottomPx = { x: (1 - hiltBottom.x) * width, y: hiltBottom.y * height };
      const topPx = { x: (1 - hiltTop.x) * width, y: hiltTop.y * height };

      const gradient = ctx.createLinearGradient(bottomPx.x, bottomPx.y, topPx.x, topPx.y);
      gradient.addColorStop(0, '#2a2a2a'); 
      gradient.addColorStop(0.2, '#888');   
      gradient.addColorStop(1, '#000');     
      ctx.lineWidth = 20; 
      ctx.strokeStyle = gradient;
      ctx.beginPath();
      ctx.moveTo(bottomPx.x, bottomPx.y);
      ctx.lineTo(topPx.x, topPx.y);
      ctx.stroke();

      if (swordState === GameState.HILT_FORMING && progress > 0) {
           ctx.shadowBlur = 20 * progress;
           ctx.shadowColor = color;
           ctx.fillStyle = color;
           ctx.beginPath();
           ctx.arc(topPx.x, topPx.y, 10 + 20 * progress, 0, Math.PI * 2);
           ctx.fill();
           ctx.shadowBlur = 0;
      }

      let length = maxLength;
      if (swordState === GameState.HILT_FORMING) length = 0;
      else if (swordState === GameState.CASTING || swordState === GameState.DISSOLVING) length *= progress;

      if (length > 0.01) {
          const tipPosNormalized = addVector(hiltTop, scaleVector(direction, length));
          const tipPx = { x: (1 - tipPosNormalized.x) * width, y: tipPosNormalized.y * height };
          const hiltTopPx = { x: (1 - hiltTop.x) * width, y: hiltTop.y * height };

          if (swordState === GameState.ACTIVE) {
              trail.unshift({ tip: { ...tipPx }, base: { ...hiltTopPx }, color });
              if (trail.length > 12) trail.pop();
              if (trail.length > 1) {
                  ctx.save();
                  ctx.globalCompositeOperation = 'screen';
                  for (let i = 0; i < trail.length - 1; i++) {
                      const curr = trail[i];
                      const next = trail[i+1];
                      ctx.beginPath();
                      ctx.moveTo(curr.tip.x, curr.tip.y);
                      ctx.lineTo(next.tip.x, next.tip.y);
                      ctx.lineTo(next.base.x, next.base.y);
                      ctx.lineTo(curr.base.x, curr.base.y);
                      ctx.closePath();
                      ctx.fillStyle = `rgba(255, 255, 255, ${(1 - (i / trail.length)) * 0.3})`;
                      ctx.fill();
                  }
                  ctx.restore();
              }
          } else {
              trail.length = 0;
          }

          ctx.lineCap = 'round';
          ctx.shadowBlur = 35;
          ctx.shadowColor = color;
          ctx.strokeStyle = color;
          ctx.lineWidth = 15;
          ctx.globalCompositeOperation = 'screen';
          ctx.beginPath();
          ctx.moveTo(hiltTopPx.x, hiltTopPx.y);
          ctx.lineTo(tipPx.x, tipPx.y);
          ctx.stroke();

          ctx.shadowBlur = 10;
          ctx.shadowColor = '#fff';
          ctx.strokeStyle = '#fff';
          ctx.lineWidth = 8;
          ctx.beginPath();
          ctx.moveTo(hiltTopPx.x, hiltTopPx.y);
          ctx.lineTo(tipPx.x, tipPx.y);
          ctx.stroke();

          ctx.globalCompositeOperation = 'source-over';
          ctx.shadowBlur = 0;

          return { p1: hiltTopPx, p2: tipPx };
      }
      return null;
  };

  const drawLoop = useCallback(() => {
    const video = webcamRef.current?.video;
    const canvas = canvasRef.current;
    
    if (video && video.readyState === 4 && canvas && handLandmarkerRef.current) {
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const now = Date.now();
      const w = canvas.width;
      const h = canvas.height;

      const handResults = handLandmarkerRef.current.detectForVideo(video, now);
      const hands: HandData[] = [];
      if (handResults.landmarks) {
        handResults.landmarks.forEach((landmarks, index) => {
          const handedness = handResults.handedness[index][0].categoryName as 'Left' | 'Right';
          hands.push({ landmarks, handedness, worldLandmarks: handResults.worldLandmarks[index] });
        });
      }

      ctx.clearRect(0, 0, w, h);

      if (gameMode === GameMode.SOLO) {
          if (faceLandmarkerRef.current) {
              const faceResults = faceLandmarkerRef.current.detectForVideo(video, now);
              if (faceResults.faceLandmarks?.[0]) {
                  detectSoloNod(faceResults.faceLandmarks[0][1]);
              }
          }
          processSoloPlayer(hands);
          let swordSegment = null;
          hands.forEach(hand => {
              const isSwordHand = hand.handedness === swordHandRef.current;
              drawSkeleton(ctx, hand, w, h, isSwordHand ? 'rgba(0, 255, 255, 0.6)' : 'rgba(255, 200, 0, 0.6)');
              if (isSwordHand) {
                  swordSegment = drawSaber(
                      ctx, hand, gameStateRef.current, swordProgressRef.current, 
                      velocityRef.current, trailHistoryRef.current, '#00aaff', w, h,
                      SWORD_LENGTH_SOLO
                  );
                  const { hiltTop, direction } = getGripPoints(hand);
                  let len = SWORD_LENGTH_SOLO;
                  if (gameStateRef.current === GameState.CASTING) len *= swordProgressRef.current;
                  const tip = addVector(hiltTop, scaleVector(direction, len));
                  if (gameStateRef.current === GameState.ACTIVE || gameStateRef.current === GameState.CASTING) {
                      swordSegment = { 
                          p1: { x: 1 - hiltTop.x, y: hiltTop.y }, 
                          p2: { x: 1 - tip.x, y: tip.y } 
                      };
                  } else {
                      swordSegment = null;
                  }
              }
          });
          updateSoloGameLogic(w, h, swordSegment);

      } else {
          // --- DUEL MODE ---
          let faces: any[] = [];
          if (faceLandmarkerRef.current) {
              const res = faceLandmarkerRef.current.detectForVideo(video, now);
              if (res.faceLandmarks) faces = res.faceLandmarks.map((l, i) => [i, l[1], l]);
          }

          let p1Face = null;
          let p2Face = null;
          if (faces.length > 0) {
              faces.sort((a,b) => b[1].x - a[1].x);
              p1Face = faces[0];
              if (faces.length > 1) p2Face = faces[1];
          }

          // Game Start Check (New: Both Swords Active)
          if (duelStateRef.current === GameState.DUEL_WAITING) {
              if (player1Ref.current.swordState === GameState.ACTIVE && player2Ref.current.swordState === GameState.ACTIVE) {
                  duelStateRef.current = GameState.DUEL_FIGHTING;
                  spawnParticles(w * 0.25, h * 0.5, 30, 'shockwave', '#00ffff', 0.04, 5);
                  spawnParticles(w * 0.75, h * 0.5, 30, 'shockwave', '#ff3333', 0.04, 5);
                  player1Ref.current.hp = 100;
                  player2Ref.current.hp = 100;
                  setP1Hp(100);
                  setP2Hp(100);
              }
          }

          const p1Hands: HandData[] = [];
          const p2Hands: HandData[] = [];
          hands.forEach(hand => {
             const center = getHandCenter(hand);
             if (p1Face && p2Face) {
                 const d1 = getDistance(center, p1Face[1]);
                 const d2 = getDistance(center, p2Face[1]);
                 if (d1 < d2) p1Hands.push(hand);
                 else p2Hands.push(hand);
             } else {
                 // Fallback: Screen Left = P1, Screen Right = P2
                 if (center.x > 0.5) p1Hands.push(hand);
                 else p2Hands.push(hand);
             }
          });

          processDuelPlayer(player1Ref.current, p1Hands);
          processDuelPlayer(player2Ref.current, p2Hands);

          const renderPlayer = (p: DuelPlayerState, myHands: HandData[], color: string) => {
              let seg = null;
              myHands.forEach(hand => {
                  const isSword = hand.handedness === p.swordHand;
                  drawSkeleton(ctx, hand, w, h, isSword ? color : '#aaa');
                  if (isSword) {
                     seg = drawSaber(
                         ctx, hand, p.swordState, p.swordProgress, p.velocity, 
                         p.trail, p.saberColor, w, h, SWORD_LENGTH_DUEL
                     );
                  }
              });
              return seg;
          };

          const s1 = renderPlayer(player1Ref.current, p1Hands, '#00ffff');
          const s2 = renderPlayer(player2Ref.current, p2Hands, '#ff3333');

          if (duelStateRef.current === GameState.DUEL_FIGHTING) {
              if (s1 && s2) {
                  let collisionPt: Vector2 | null = getLineIntersection(s1.p1, s1.p2, s2.p1, s2.p2);
                  
                  if (!collisionPt) {
                      const d1 = distToSegment(s1.p2, s2.p1, s2.p2); 
                      const d2 = distToSegment(s2.p2, s1.p1, s1.p2); 
                      const THRESHOLD = 50; 
                      if (d1 < THRESHOLD) collisionPt = s1.p2;
                      else if (d2 < THRESHOLD) collisionPt = s2.p2;
                  }

                  if (collisionPt) {
                      setClashFlash(true);
                      setTimeout(() => setClashFlash(false), 50);

                      spawnParticles(collisionPt.x / w, collisionPt.y / h, 8, 'spark', '#ffffff', 0.04, 7);
                      spawnParticles(collisionPt.x / w, collisionPt.y / h, 2, 'shockwave', '#ffff00', 0.03, 15);
                  }
              }

              const checkHit = (saberSeg: {p1: Vector2, p2: Vector2} | null, targetFace: any, targetPlayer: DuelPlayerState, isRightSidePlayer: boolean) => {
                 if (!saberSeg || now - targetPlayer.lastHitTime < 500) return; 
                 
                 let boxX, boxY, boxW, boxH;

                 if (targetFace) {
                     const nose = targetFace[1];
                     const nx = (1 - nose.x) * w;
                     const ny = nose.y * h;
                     boxW = 300;
                     boxH = 700;
                     boxX = nx - boxW/2;
                     boxY = ny - 150;
                 } else {
                     // Fallback Hitbox if face is lost
                     // P1 (Left) -> x ~ 25% w
                     // P2 (Right) -> x ~ 75% w
                     const centerX = isRightSidePlayer ? w * 0.75 : w * 0.25;
                     const centerY = h * 0.6;
                     boxW = 350;
                     boxH = 800;
                     boxX = centerX - boxW/2;
                     boxY = h * 0.2; // From top area down
                 }

                 if (lineRectCollide(saberSeg.p1.x, saberSeg.p1.y, saberSeg.p2.x, saberSeg.p2.y, boxX, boxY, boxW, boxH)) {
                     targetPlayer.hp = Math.max(0, targetPlayer.hp - 8);
                     targetPlayer.lastHitTime = now;
                     
                     setHitFlash(targetPlayer.id);
                     setTimeout(() => setHitFlash(0), 100);

                     spawnParticles(0.5, 0.5, 30, 'spark', '#ff0000', 0.08, 6); 
                     
                     if (targetPlayer.hp <= 0) {
                         duelStateRef.current = GameState.DUEL_ENDED;
                         setDuelWinner(targetPlayer.id === 1 ? 2 : 1);
                     } else {
                         if (targetPlayer.id === 1) setP1Hp(targetPlayer.hp);
                         else setP2Hp(targetPlayer.hp);
                     }
                 }
              };

              // Check P2 hit (P1 attacking P2) -> P2 is Right Side
              checkHit(s1, p2Face, player2Ref.current, true);
              // Check P1 hit (P2 attacking P1) -> P1 is Left Side
              checkHit(s2, p1Face, player1Ref.current, false);
          }
      }

      if (gameMode === GameMode.SOLO) {
          ctx.font = 'bold 30px sans-serif';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          enemiesRef.current.forEach(enemy => {
              const renderX = (1 - enemy.x) * w;
              const renderY = enemy.y * h;
              ctx.fillStyle = '#ffffff';
              ctx.shadowBlur = 4;
              ctx.shadowColor = 'rgba(255, 100, 100, 0.8)';
              ctx.fillText(enemy.text, renderX, renderY);
              ctx.shadowBlur = 0;
          });
      }

      for (let i = particlesRef.current.length - 1; i >= 0; i--) {
        const p = particlesRef.current[i];
        p.life -= 0.02;
        p.x += p.vx;
        p.y += p.vy;
        if (p.life <= 0) { particlesRef.current.splice(i, 1); continue; }
        const x = (1 - p.x) * w;
        const y = p.y * h;
        const alpha = p.life / p.maxLife;
        ctx.globalAlpha = alpha;
        ctx.fillStyle = p.color;
        ctx.beginPath();
        const currentSize = p.size * (0.5 + 0.5 * alpha);
        ctx.arc(x, y, currentSize, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1.0;
      }
    }
    
    requestRef.current = requestAnimationFrame(drawLoop);
  }, [gameMode]);

  useEffect(() => {
    requestRef.current = requestAnimationFrame(drawLoop);
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [drawLoop]);

  return (
    <div className={`relative w-full h-screen flex justify-center items-center overflow-hidden bg-black ${hitFlash !== 0 ? 'animate-shake' : ''}`}>
      <style>{`
        @keyframes shake {
          0% { transform: translate(1px, 1px) rotate(0deg); }
          10% { transform: translate(-1px, -2px) rotate(-1deg); }
          20% { transform: translate(-3px, 0px) rotate(1deg); }
          30% { transform: translate(3px, 2px) rotate(0deg); }
          40% { transform: translate(1px, -1px) rotate(1deg); }
          50% { transform: translate(-1px, 2px) rotate(-1deg); }
          60% { transform: translate(-3px, 1px) rotate(0deg); }
          70% { transform: translate(3px, 1px) rotate(-1deg); }
          80% { transform: translate(-1px, -1px) rotate(1deg); }
          90% { transform: translate(1px, 2px) rotate(0deg); }
          100% { transform: translate(1px, -2px) rotate(-1deg); }
        }
        .animate-shake { animation: shake 0.5s; }
      `}</style>
      <Webcam
        ref={webcamRef}
        className="absolute w-full h-full object-cover transform scale-x-[-1]"
        videoConstraints={{ width: 1280, height: 720, facingMode: "user" }}
      />
      <canvas ref={canvasRef} className="absolute w-full h-full pointer-events-none" />
      
      {/* HIT FLASH OVERLAY */}
      {hitFlash === 1 && <div className="absolute inset-y-0 left-0 w-1/2 bg-red-500/40 pointer-events-none transition-opacity duration-100"></div>}
      {hitFlash === 2 && <div className="absolute inset-y-0 right-0 w-1/2 bg-red-500/40 pointer-events-none transition-opacity duration-100"></div>}
      {clashFlash && <div className="absolute inset-0 bg-white/30 pointer-events-none transition-opacity duration-75"></div>}

      {/* GLOBAL UI */}
      <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-20 flex gap-4">
          <button 
             onClick={toggleMode}
             className={`px-4 py-2 rounded-full font-bold text-white border transition-all ${gameMode === GameMode.SOLO ? 'bg-cyan-600 border-cyan-400' : 'bg-red-600 border-red-400'}`}
          >
             MODE: {gameMode === GameMode.SOLO ? 'SOLO SURVIVAL' : 'DUAL DUEL'}
          </button>
      </div>

      {loading && (
        <div className="absolute z-50 text-white text-2xl font-sans animate-pulse">Initializing Systems...</div>
      )}

      {/* SOLO HUD */}
      {!loading && gameMode === GameMode.SOLO && (
          <>
            <div className="absolute top-4 right-4 text-white font-sans text-xl z-10 font-bold tracking-wider">SCORE: {score}</div>
            {!isGameActive && gameState === GameState.ACTIVE && (
                <div className="absolute top-1/4 left-1/2 transform -translate-x-1/2 text-center pointer-events-none">
                    <h2 className="text-3xl text-white font-bold mb-2 animate-bounce drop-shadow-md">NOD to Start</h2>
                </div>
            )}
            {gameState === GameState.IDLE && (
                <div className="absolute top-20 text-center text-white font-sans bg-gray-900/80 p-6 rounded-2xl border border-white/10 backdrop-blur-md">
                   <h2 className="text-xl font-bold">Solo Mode</h2>
                   <p className="text-sm">Make a FIST to summon hilt.</p>
                </div>
            )}
          </>
      )}

      {/* DUEL HUD */}
      {!loading && gameMode === GameMode.DUEL && (
          <>
             <div className="absolute top-20 left-10 w-64">
                 <div className="text-cyan-400 font-bold mb-1">PLAYER 1 (LEFT)</div>
                 <div className="h-4 w-full bg-gray-800 rounded overflow-hidden border border-cyan-500/50">
                     <div className="h-full bg-cyan-500 transition-all duration-300" style={{ width: `${p1Hp}%` }}></div>
                 </div>
             </div>
             <div className="absolute top-20 right-10 w-64 text-right">
                 <div className="text-red-400 font-bold mb-1">PLAYER 2 (RIGHT)</div>
                 <div className="h-4 w-full bg-gray-800 rounded overflow-hidden border border-red-500/50 flex justify-end">
                     <div className="h-full bg-red-500 transition-all duration-300" style={{ width: `${p2Hp}%` }}></div>
                 </div>
             </div>

             <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-center pointer-events-none z-10">
                 {duelStateRef.current === GameState.DUEL_WAITING && (
                     <div className="text-white text-2xl font-bold animate-pulse bg-black/50 p-4 rounded text-center">
                         BOTH PLAYERS:<br/>
                         HOLD FIST TO IGNITE SWORDS!
                     </div>
                 )}
                 {duelStateRef.current === GameState.DUEL_FIGHTING && (
                     <div className="text-white/50 text-sm bg-black/30 p-2 rounded">
                         FIGHT!
                     </div>
                 )}
                 {duelStateRef.current === GameState.DUEL_ENDED && (
                     <div className="bg-black/80 p-8 rounded-xl border-2 border-white">
                         <div className={`text-6xl font-black mb-4 ${duelWinner === 1 ? 'text-cyan-400' : 'text-red-500'}`}>
                             {duelWinner === 1 ? 'PLAYER 1' : 'PLAYER 2'} WINS!
                         </div>
                         <div className="text-white text-xl">Press Mode Button to Reset</div>
                     </div>
                 )}
             </div>
          </>
      )}
    </div>
  );
};

export default LightsaberAR;