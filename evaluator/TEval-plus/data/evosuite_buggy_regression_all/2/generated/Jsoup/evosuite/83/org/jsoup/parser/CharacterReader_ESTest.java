/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:14:11 GMT 2023
 */

package org.jsoup.parser;

import org.junit.Test;
import static org.junit.Assert.*;
import java.io.StringReader;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.parser.CharacterReader;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CharacterReader_ESTest extends CharacterReader_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Must be false");
      characterReader0.mark();
      assertEquals(0, characterReader0.pos());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      characterReader0.advance();
      assertEquals(1, characterReader0.pos());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("]m9e0cb7Rj1Y_}1Dfi");
      characterReader0.unconsume();
      assertEquals((-1), characterReader0.pos());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("8n]} Y7r}q*yE8Z>");
      String string0 = characterReader0.toString();
      assertEquals(0, characterReader0.pos());
      assertEquals("8n]} Y7r}q*yE8Z>", string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("2Sa+ni&y,=W7h.wDYt");
      boolean boolean0 = characterReader0.rangeEquals((-19), (-19), "2Sa+ni&y,=W7h.wDYt");
      assertFalse(boolean0);
      assertEquals(0, characterReader0.pos());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      StringReader stringReader0 = new StringReader(".z%rD]LsW;X`h");
      CharacterReader characterReader0 = new CharacterReader(stringReader0);
      int int0 = characterReader0.pos();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      StringReader stringReader0 = new StringReader("5=!~");
      CharacterReader characterReader0 = new CharacterReader(stringReader0, 65535);
      assertEquals(0, characterReader0.pos());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("8n]} Y7r}q*yE8Z>");
      boolean boolean0 = characterReader0.matchesLetter();
      assertEquals(0, characterReader0.pos());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      char[] charArray0 = new char[0];
      boolean boolean0 = characterReader0.matchesAnySorted(charArray0);
      assertEquals(0, characterReader0.pos());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("8n]} Y7r}q*yE8Z>");
      char char0 = characterReader0.current();
      assertEquals('8', char0);
      assertEquals(0, characterReader0.pos());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("8n]} Y7r}q*yE8Z>");
      characterReader0.consumeData();
      characterReader0.consumeLetterThenDigitSequence();
      assertEquals(16, characterReader0.pos());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      char char0 = characterReader0.current();
      assertEquals(0, characterReader0.pos());
      assertEquals('\uFFFF', char0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("'");
      char char0 = characterReader0.consume();
      assertEquals(1, characterReader0.pos());
      assertEquals('\'', char0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      char char0 = characterReader0.consume();
      assertEquals(1, characterReader0.pos());
      assertEquals('\uFFFF', char0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Array must not contain any null bject");
      characterReader0.consumeTo(';');
      assertEquals(37, characterReader0.pos());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("q!U4]J$tdXz{G$$?");
      characterReader0.consumeTo('d');
      assertEquals(8, characterReader0.pos());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("W):wP0/'i.-7s");
      characterReader0.consumeTo("org.jsoup.parser.CharacterReader");
      assertEquals(13, characterReader0.pos());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Array must not contain any null objects");
      characterReader0.consumeTo("Array must not contain any null objects");
      assertEquals(0, characterReader0.pos());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("org.jsoup.parser.CharacterReader");
      boolean boolean0 = characterReader0.containsIgnoreCase("org.jsoup.parser.CharacterReader");
      assertEquals(0, characterReader0.pos());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      char[] charArray0 = new char[0];
      characterReader0.consumeToAny(charArray0);
      assertEquals(0, characterReader0.pos());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      StringReader stringReader0 = new StringReader("&b){)~<zDVy2<$z2");
      CharacterReader characterReader0 = new CharacterReader(stringReader0);
      char[] charArray0 = new char[7];
      characterReader0.consumeToAnySorted(charArray0);
      characterReader0.matches('W');
      assertEquals(17, characterReader0.pos());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("org.jsoup.UncheckedIOException");
      char[] charArray0 = new char[2];
      charArray0[0] = 'p';
      characterReader0.consumeToAnySorted(charArray0);
      assertEquals(8, characterReader0.pos());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("org.jsoup.parser.CharacterReader");
      characterReader0.consumeData();
      char[] charArray0 = new char[2];
      characterReader0.consumeToAnySorted(charArray0);
      assertEquals(32, characterReader0.pos());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader(":Q{rRcSN&rQuWA");
      characterReader0.consumeData();
      assertEquals(8, characterReader0.pos());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      StringReader stringReader0 = new StringReader("&b){)~<zDVy2<$z2");
      CharacterReader characterReader0 = new CharacterReader(stringReader0);
      char[] charArray0 = new char[7];
      charArray0[0] = '~';
      characterReader0.consumeToAny(charArray0);
      characterReader0.consumeData();
      assertEquals(6, characterReader0.pos());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("8n]} Y7r}q*yE8Z>");
      characterReader0.consumeData();
      characterReader0.consumeData();
      assertEquals(16, characterReader0.pos());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("8n]} Y7r}q*yE8Z>");
      characterReader0.consumeData();
      characterReader0.consumeTagName();
      assertEquals(16, characterReader0.pos());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("*~6?o\"/YNkdP*J");
      characterReader0.consumeTagName();
      assertEquals(6, characterReader0.pos());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("s!>Bc;7P");
      characterReader0.consumeTagName();
      assertEquals(2, characterReader0.pos());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Array must @ot contain any nu~l objects");
      characterReader0.consumeData();
      characterReader0.consumeLetterSequence();
      assertEquals(39, characterReader0.pos());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Arrxy mst Nt o3tain ady nullA{bjects");
      characterReader0.consumeLetterSequence();
      assertEquals(5, characterReader0.pos());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      StringReader stringReader0 = new StringReader("&b){)~<zDVy2<$z2");
      CharacterReader characterReader0 = new CharacterReader(stringReader0);
      char[] charArray0 = new char[7];
      charArray0[0] = '~';
      characterReader0.consumeToAny(charArray0);
      characterReader0.consumeLetterSequence();
      assertEquals(5, characterReader0.pos());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("d~/=6zWx");
      characterReader0.consumeLetterThenDigitSequence();
      assertEquals(1, characterReader0.pos());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Sd80z%SH 9=P-WOq");
      characterReader0.consumeLetterThenDigitSequence();
      assertEquals(4, characterReader0.pos());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      characterReader0.consumeHexSequence();
      assertEquals(0, characterReader0.pos());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("8n]} Y7r}q*yE8Z>");
      String string0 = characterReader0.consumeHexSequence();
      assertEquals("8", string0);
      
      boolean boolean0 = characterReader0.matchesLetter();
      assertEquals(1, characterReader0.pos());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("*~6?o\"/YNkdP*J");
      characterReader0.consumeHexSequence();
      assertEquals(0, characterReader0.pos());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Aray m'su not c`ntain any null objects");
      characterReader0.consumeHexSequence();
      assertEquals(1, characterReader0.pos());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("eSNxVyyEA`");
      String string0 = characterReader0.consumeHexSequence();
      assertEquals(1, characterReader0.pos());
      assertEquals("e", string0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      characterReader0.consumeDigitSequence();
      assertEquals(0, characterReader0.pos());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("8n]} Y7r}q*yE8Z>");
      characterReader0.consumeDigitSequence();
      assertEquals(1, characterReader0.pos());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("\"mGj-\"FMxm!R3");
      String string0 = characterReader0.consumeDigitSequence();
      assertEquals("", string0);
      assertEquals(0, characterReader0.pos());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Array must not contain any null objects");
      boolean boolean0 = characterReader0.matches('-');
      assertEquals(0, characterReader0.pos());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("-pseDC 't]C`<3dh~h");
      boolean boolean0 = characterReader0.matches('-');
      assertEquals(0, characterReader0.pos());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Array must no contain any nullobjects");
      boolean boolean0 = characterReader0.matchConsume("a{{y<*");
      assertEquals(0, characterReader0.pos());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Must be fals");
      characterReader0.consumeTagName();
      boolean boolean0 = characterReader0.matches("Must be fals");
      assertEquals(4, characterReader0.pos());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("{M%CMe*IOrQ5W=O3*p");
      boolean boolean0 = characterReader0.matchConsume("{M%CMe*IOrQ5W=O3*p");
      assertEquals(19, characterReader0.pos());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Array must not contain any null objects");
      boolean boolean0 = characterReader0.matchConsumeIgnoreCase("n</4-W/;(P");
      assertEquals(0, characterReader0.pos());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("\"mGj-\"FMxm!R3");
      characterReader0.consumeToEnd();
      boolean boolean0 = characterReader0.matchConsumeIgnoreCase("\"mGj-\"FMxm!R3");
      assertEquals(13, characterReader0.pos());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("m9e0cb7Rj1Y_}1Dfi");
      boolean boolean0 = characterReader0.matchConsumeIgnoreCase("m9e0cb7Rj1Y_}1Dfi");
      assertEquals(17, characterReader0.pos());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Array must @ot contain any nu~l objects");
      characterReader0.consumeData();
      characterReader0.matchesAny((char[]) null);
      assertEquals(39, characterReader0.pos());
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("#P-FrEK17U");
      char[] charArray0 = new char[3];
      charArray0[0] = '%';
      boolean boolean0 = characterReader0.matchesAny(charArray0);
      assertFalse(boolean0);
      assertEquals(0, characterReader0.pos());
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("^fMP%[z8t x?R");
      char[] charArray0 = new char[1];
      boolean boolean0 = characterReader0.matchesAnySorted(charArray0);
      assertEquals(0, characterReader0.pos());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("^fMP%[z8t x?R");
      char[] charArray0 = new char[1];
      charArray0[0] = '^';
      boolean boolean0 = characterReader0.matchesAnySorted(charArray0);
      assertEquals(0, characterReader0.pos());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      boolean boolean0 = characterReader0.matchesLetter();
      assertEquals(0, characterReader0.pos());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Must be false");
      boolean boolean0 = characterReader0.matchesLetter();
      assertEquals(0, characterReader0.pos());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("~IToBr-YYgdswwB");
      boolean boolean0 = characterReader0.matchesLetter();
      assertFalse(boolean0);
      assertEquals(0, characterReader0.pos());
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("!9eDr,TQ");
      boolean boolean0 = characterReader0.matchesDigit();
      assertEquals(0, characterReader0.pos());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      boolean boolean0 = characterReader0.matchesDigit();
      assertEquals(0, characterReader0.pos());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("8KazbcvZP6'U1^Sn");
      boolean boolean0 = characterReader0.matchesDigit();
      assertEquals(0, characterReader0.pos());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Array must not contain any null objects");
      boolean boolean0 = characterReader0.matchesDigit();
      assertEquals(0, characterReader0.pos());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("rray must not contain any n2~l objects");
      boolean boolean0 = characterReader0.containsIgnoreCase("rray must not contain any n2~l objects");
      assertEquals(0, characterReader0.pos());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("5|)N");
      boolean boolean0 = characterReader0.containsIgnoreCase("5|)N");
      assertTrue(boolean0);
      assertEquals(0, characterReader0.pos());
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("8n]} Y7r}q*yE8Z>");
      characterReader0.consumeHexSequence();
      characterReader0.consumeTagName();
      characterReader0.consumeData();
      assertEquals(16, characterReader0.pos());
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Arrxy mst Nt o3tain ady nullA{bjects");
      characterReader0.consumeTagName();
      characterReader0.rewindToMark();
      characterReader0.consumeLetterThenDigitSequence();
      assertEquals(5, characterReader0.pos());
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      char[] charArray0 = new char[7];
      boolean boolean0 = CharacterReader.rangeEquals(charArray0, 1, 15, "6z KEzKxMQ>CyMX");
      assertFalse(boolean0);
  }
}