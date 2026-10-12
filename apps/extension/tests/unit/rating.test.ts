import { describe, it, expect } from "vitest";
import { detectStore, ratingAskPending, recordSuccessfulRead, settleRatingAsk, RATING_ASK_AT } from "../../src/rating";

function mem(): Storage {
  const m = new Map<string, string>();
  return { getItem: (k) => m.get(k) ?? null, setItem: (k, v) => void m.set(k, v) } as Storage;
}

describe("rating ask", () => {
  it("fires exactly on the Nth successful read, and stays pending until answered", () => {
    const s = mem();
    for (let i = 1; i < RATING_ASK_AT; i++) {
      expect(recordSuccessfulRead(s)).toBe(false);
      expect(ratingAskPending(s)).toBe(false);
    }
    expect(recordSuccessfulRead(s)).toBe(true);
    expect(ratingAskPending(s)).toBe(true);
    expect(recordSuccessfulRead(s)).toBe(false); // the 6th read does not re-fire
    expect(ratingAskPending(s)).toBe(true); // but the ask is still up until answered
  });

  it("never shows again once answered, either way", () => {
    const s = mem();
    for (let i = 0; i < RATING_ASK_AT; i++) recordSuccessfulRead(s);
    settleRatingAsk(s);
    expect(ratingAskPending(s)).toBe(false);
    for (let i = 0; i < 20; i++) expect(recordSuccessfulRead(s)).toBe(false);
    expect(ratingAskPending(s)).toBe(false);
  });

  it("survives a storage that throws (private mode): no ask, no crash", () => {
    const s = { getItem: () => { throw new Error("blocked"); }, setItem: () => { throw new Error("blocked"); } } as unknown as Storage;
    expect(recordSuccessfulRead(s)).toBe(false);
    expect(ratingAskPending(s)).toBe(false);
  });

  it("links to the store the user installed from", () => {
    expect(detectStore("Mozilla/5.0 ... Chrome/128 Safari/537.36 Edg/128.0")).toBe("edge");
    expect(detectStore("Mozilla/5.0 (X11; Linux x86_64; rv:130.0) Gecko/20100101 Firefox/130.0")).toBe("firefox");
    expect(detectStore("Mozilla/5.0 ... Chrome/128 Safari/537.36")).toBe("chrome");
  });
});
