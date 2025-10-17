
type GeoPoint = {
  name?: string
  lat: number
  lon: number
}

export interface LeafletMapProps {
  from?: GeoPoint | null
  to?: GeoPoint | null
  animateKey?: string | number
  isPredicting?: boolean
}

export interface RouteStep {
  instruction: {
    text: string
  }
}
