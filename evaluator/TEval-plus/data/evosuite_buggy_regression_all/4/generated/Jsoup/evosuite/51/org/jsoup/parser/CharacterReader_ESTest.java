/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:54:13 GMT 2023
 */

package org.jsoup.parser;

import org.junit.Test;
import static org.junit.Assert.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.parser.CharacterReader;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CharacterReader_ESTest extends CharacterReader_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("X?M\"/lk{)6/");
      characterReader0.mark();
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Must be false");
      characterReader0.advance();
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("(.C1k2a Z");
      characterReader0.unconsume();
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Must be false");
      String string0 = characterReader0.toString();
      assertEquals("Must be false", string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("UypP+&O!jYwb");
      String string0 = characterReader0.consumeAsString();
      assertEquals("U", string0);
      
      String string1 = characterReader0.consumeTo("UypP+&O!jYwb");
      assertEquals("ypP+&O!jYwb", string1);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Vject mustenot be null");
      String string0 = characterReader0.consumeTo('Z');
      assertEquals("Vject mustenot be null", string0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("mm=mkFKG}@/~]$(");
      int int0 = characterReader0.pos();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("{!\" 6dC+O");
      String string0 = characterReader0.consumeLetterThenDigitSequence();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      String string0 = characterReader0.consumeLetterThenDigitSequence();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("j");
      char char0 = characterReader0.current();
      assertEquals('j', char0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      char char0 = characterReader0.current();
      assertEquals('\uFFFF', char0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("UypPz+&O!jYwb");
      char char0 = characterReader0.consume();
      assertEquals('U', char0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      char char0 = characterReader0.consume();
      assertEquals('\uFFFF', char0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("!2G?\"$ 6+2@");
      String string0 = characterReader0.consumeTo('2');
      assertEquals("!", string0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("dxuLkS;bTJs?!vrD<&");
      boolean boolean0 = characterReader0.containsIgnoreCase("dxuLkS;bTJs?!vrD<&");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader(">7P\"$ 4+2@");
      boolean boolean0 = characterReader0.containsIgnoreCase(">7P\"$ 4+2@");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("bjct mu=l nt be&null");
      String string0 = characterReader0.consumeTo("bjct mu=l nt be&null");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      char[] charArray0 = new char[1];
      String string0 = characterReader0.consumeToAny(charArray0);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("String must not be empty");
      char[] charArray0 = new char[1];
      String string0 = characterReader0.consumeToAny(charArray0);
      assertEquals("String must not be empty", string0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("w.I_.:x,Cya)`pE=456");
      char[] charArray0 = new char[3];
      charArray0[0] = 'p';
      String string0 = characterReader0.consumeToAny(charArray0);
      assertEquals("w.I_.:x,Cya)`", string0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("us} e 8ls");
      char[] charArray0 = new char[1];
      String string0 = characterReader0.consumeToAnySorted(charArray0);
      assertEquals("us} e 8ls", string0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("us} e 8ls");
      char[] charArray0 = new char[1];
      charArray0[0] = 'u';
      String string0 = characterReader0.consumeToAnySorted(charArray0);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("@2Gv3^?& Ft#t4");
      String string0 = characterReader0.consumeData();
      assertEquals("@2Gv3^?", string0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("dxuLkS;bTJs?!vrD<&");
      String string0 = characterReader0.consumeData();
      assertEquals("dxuLkS;bTJs?!vrD", string0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      String string0 = characterReader0.consumeData();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("}3)CW/U#\"\"gSa[Q");
      String string0 = characterReader0.consumeTagName();
      assertEquals("}3)CW", string0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("v6+w8>C!d!aD");
      String string0 = characterReader0.consumeTagName();
      assertEquals("v6+w8", string0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Object must not be null");
      String string0 = characterReader0.consumeLetterThenDigitSequence();
      assertEquals("Object", string0);
      
      String string1 = characterReader0.consumeTagName();
      assertEquals("", string1);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      String string0 = characterReader0.consumeLetterSequence();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Object must not be null");
      String string0 = characterReader0.consumeLetterSequence();
      assertEquals("Object", string0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("{!\" 6dC+O");
      String string0 = characterReader0.consumeLetterSequence();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("4V#(R9;}");
      String string0 = characterReader0.consumeLetterThenDigitSequence();
      assertEquals("4", string0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      String string0 = characterReader0.consumeHexSequence();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("dy");
      String string0 = characterReader0.consumeHexSequence();
      assertEquals("d", string0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("':\"TU)h<P~^~Q{=7GYV");
      String string0 = characterReader0.consumeHexSequence();
      assertEquals("", string0);
      
      String string1 = characterReader0.consumeHexSequence();
      assertEquals("", string1);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("5P-`nO");
      String string0 = characterReader0.consumeHexSequence();
      assertEquals("5", string0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("CDuw8VsHu%4=T2X[");
      String string0 = characterReader0.consumeHexSequence();
      assertEquals("CD", string0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      String string0 = characterReader0.consumeDigitSequence();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("#>cFx5");
      String string0 = characterReader0.consumeDigitSequence();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("4~S01&qb(^*@ZM8vqu");
      String string0 = characterReader0.consumeDigitSequence();
      assertEquals("4", string0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      boolean boolean0 = characterReader0.matches(']');
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("CR{je{g");
      boolean boolean0 = characterReader0.matches('\\');
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("`~.iIV837?4aoaRWI");
      boolean boolean0 = characterReader0.matchConsume("D~hw\"x");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("rD=");
      String string0 = characterReader0.consumeTagName();
      assertEquals("rD=", string0);
      
      boolean boolean0 = characterReader0.matches("rD=");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("!2?\"$ 6+2@");
      boolean boolean0 = characterReader0.matchConsumeIgnoreCase("!2?\"$ 6+2@");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("!2?\"$ 6+2@");
      boolean boolean0 = characterReader0.matchConsumeIgnoreCase("\"U1HO9K}BF3YY,:3=N");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("String must not be empty");
      boolean boolean0 = characterReader0.matchConsumeIgnoreCase("}>z");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("dy");
      char[] charArray0 = new char[3];
      boolean boolean0 = characterReader0.matchesAny(charArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      char[] charArray0 = new char[1];
      boolean boolean0 = characterReader0.matchesAny(charArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("dy");
      char[] charArray0 = new char[3];
      charArray0[2] = 'd';
      boolean boolean0 = characterReader0.matchesAny(charArray0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      char[] charArray0 = new char[0];
      boolean boolean0 = characterReader0.matchesAnySorted(charArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader(";");
      char[] charArray0 = new char[7];
      boolean boolean0 = characterReader0.matchesAnySorted(charArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader(";");
      char[] charArray0 = new char[7];
      charArray0[4] = ';';
      charArray0[5] = 'Z';
      boolean boolean0 = characterReader0.matchesAnySorted(charArray0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("dy");
      boolean boolean0 = characterReader0.matchesLetter();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      boolean boolean0 = characterReader0.matchesLetter();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Object must not be null");
      String string0 = characterReader0.consumeTagName();
      assertEquals("Object", string0);
      
      boolean boolean0 = characterReader0.matchesLetter();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("CR{je{g");
      boolean boolean0 = characterReader0.matchesLetter();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("}3)CW/U#\"\"gSa[Q");
      boolean boolean0 = characterReader0.matchesLetter();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Must be false");
      boolean boolean0 = characterReader0.matchesDigit();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("4~S01&qb(^*@ZM8vqu");
      String string0 = characterReader0.consumeTagName();
      assertEquals("4~S01&qb(^*@ZM8vqu", string0);
      
      boolean boolean0 = characterReader0.matchesDigit();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader(",+,|^+4");
      boolean boolean0 = characterReader0.matchesDigit();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("4~S01&qb(^*@ZM8vqu");
      boolean boolean0 = characterReader0.matchesDigit();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("k\"]4+j7");
      boolean boolean0 = characterReader0.matchConsume("k\"]4+j7");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("!2?\"$ 6+2@");
      boolean boolean0 = characterReader0.containsIgnoreCase("!2?\"$ 6+2@");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("CR{je{g");
      characterReader0.consumeTagName();
      characterReader0.rewindToMark();
      String string0 = characterReader0.consumeTagName();
      assertEquals("CR{je{g", string0);
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("CR{je{g");
      boolean boolean0 = characterReader0.rangeEquals(1, 1, "|");
      assertFalse(boolean0);
  }
}