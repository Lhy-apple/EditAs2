/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:47:37 GMT 2023
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
      CharacterReader characterReader0 = new CharacterReader("E-@eD*.A8hWT&,2G1oz^l");
      characterReader0.rewindToMark();
      assertEquals("E-@eD*.A8hWT&,2G1oz^l", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("JW");
      boolean boolean0 = characterReader0.matchConsumeIgnoreCase("m");
      assertFalse(boolean0);
      assertEquals("JW", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("IQvki3|");
      characterReader0.mark();
      assertEquals("IQvki3|", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("JW");
      characterReader0.advance();
      assertEquals("W", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("B)r*r.'XA:h\"9a?Fc)");
      String string0 = characterReader0.toString();
      assertEquals("B)r*r.'XA:h\"9a?Fc)", string0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("R2d3ec1jb_bA5");
      characterReader0.consumeAsString();
      characterReader0.consume();
      characterReader0.unconsume();
      characterReader0.consumeDigitSequence();
      assertEquals("d3ec1jb_bA5", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("R2d3ec1jb_bA5");
      characterReader0.consumeTo('g');
      assertEquals("", characterReader0.toString());
      
      char char0 = characterReader0.consume();
      assertEquals('\uFFFF', char0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      int int0 = characterReader0.pos();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("IQvki3|");
      char[] charArray0 = new char[5];
      boolean boolean0 = characterReader0.matchesAny(charArray0);
      assertEquals("IQvki3|", characterReader0.toString());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("IQvki3|");
      char[] charArray0 = new char[0];
      characterReader0.consumeToAny(charArray0);
      characterReader0.consumeHexSequence();
      assertEquals("", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("%G=ti0");
      char char0 = characterReader0.current();
      assertEquals('%', char0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      char char0 = characterReader0.current();
      assertEquals('\uFFFF', char0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("R2d3ec1jb_bA5");
      char char0 = characterReader0.consume();
      assertEquals("2d3ec1jb_bA5", characterReader0.toString());
      assertEquals('R', char0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("E-@eD*.A8hWT&,2G1oz^l");
      characterReader0.consumeTo('o');
      assertEquals("oz^l", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("B)r*r.'XA:h\"9a?Fc)");
      characterReader0.consumeTo("org.jsoup.parser.CharacterReader");
      assertEquals("", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("zWsua");
      characterReader0.consumeTo("");
      assertEquals("zWsua", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("IQvki3|");
      char[] charArray0 = new char[5];
      charArray0[0] = '|';
      characterReader0.consumeToAny(charArray0);
      assertEquals("|", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("IQvki3|");
      char[] charArray0 = new char[5];
      charArray0[0] = '|';
      characterReader0.consumeLetterSequence();
      characterReader0.consumeHexSequence();
      characterReader0.consumeToAny(charArray0);
      assertEquals("|", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      String string0 = characterReader0.consumeLetterSequence();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("iRspF}\"C");
      characterReader0.consumeLetterSequence();
      assertEquals("}\"C", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("E-@eD*.A8hWT&,2G1oz^l");
      characterReader0.consumeHexSequence();
      assertEquals("-@eD*.A8hWT&,2G1oz^l", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("efp?98/z8hwt&,2g1oz^le-@ed*.a8hwt&,2g1oz^l");
      characterReader0.consumeHexSequence();
      assertEquals("p?98/z8hwt&,2g1oz^le-@ed*.a8hwt&,2g1oz^l", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("IQvki3|");
      char[] charArray0 = new char[0];
      characterReader0.consumeToAny(charArray0);
      characterReader0.consumeDigitSequence();
      assertEquals("", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("R2d3ec1jb_bA5");
      String string0 = characterReader0.consumeDigitSequence();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("%G=ti0");
      String string0 = characterReader0.consumeDigitSequence();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("SIL");
      characterReader0.consumeTo('5');
      boolean boolean0 = characterReader0.matches('5');
      assertEquals("", characterReader0.toString());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("%G=ti0");
      boolean boolean0 = characterReader0.matches('=');
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("%G=ti0");
      characterReader0.consumeAsString();
      characterReader0.consumeAsString();
      boolean boolean0 = characterReader0.matches('=');
      assertEquals("=ti0", characterReader0.toString());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("IQvki3|");
      char[] charArray0 = new char[0];
      characterReader0.consumeToAny(charArray0);
      boolean boolean0 = characterReader0.matchesAny(charArray0);
      assertEquals("", characterReader0.toString());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("IQvki3|");
      char[] charArray0 = new char[5];
      charArray0[0] = '|';
      characterReader0.consumeLetterSequence();
      characterReader0.consumeHexSequence();
      boolean boolean0 = characterReader0.matchesAny(charArray0);
      assertEquals("|", characterReader0.toString());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("IQvki3|");
      boolean boolean0 = characterReader0.matchesLetter();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("SIL");
      characterReader0.consumeTo('5');
      boolean boolean0 = characterReader0.matchesLetter();
      assertEquals("", characterReader0.toString());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("%G=ti0");
      boolean boolean0 = characterReader0.matchesLetter();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("IQvki3|");
      characterReader0.consumeLetterSequence();
      characterReader0.consumeHexSequence();
      boolean boolean0 = characterReader0.matchesLetter();
      assertEquals("|", characterReader0.toString());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("zWsua");
      boolean boolean0 = characterReader0.matchesLetter();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("%G=ti0");
      boolean boolean0 = characterReader0.matchesDigit();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      boolean boolean0 = characterReader0.matchesDigit();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("%G=ti0");
      characterReader0.matchConsumeIgnoreCase("%G=ti0");
      characterReader0.unconsume();
      boolean boolean0 = characterReader0.matchesDigit();
      assertEquals("0", characterReader0.toString());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("zWsua");
      boolean boolean0 = characterReader0.matchesDigit();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader(">m(l");
      characterReader0.consumeAsString();
      boolean boolean0 = characterReader0.matchConsume(">m(l");
      assertEquals("m(l", characterReader0.toString());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("%G=ti0");
      boolean boolean0 = characterReader0.matchConsume("");
      assertEquals("%G=ti0", characterReader0.toString());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("%G=ti0");
      boolean boolean0 = characterReader0.containsIgnoreCase("");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("%G=ti0");
      boolean boolean0 = characterReader0.containsIgnoreCase("%G=ti0");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("8GDA!");
      boolean boolean0 = characterReader0.containsIgnoreCase("8GDA!");
      assertTrue(boolean0);
  }
}
