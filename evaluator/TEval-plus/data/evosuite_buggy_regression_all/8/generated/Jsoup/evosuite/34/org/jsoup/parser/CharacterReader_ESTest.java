/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 03:28:13 GMT 2023
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
      CharacterReader characterReader0 = new CharacterReader("ryMDi5gL-WOpy[=)");
      characterReader0.rewindToMark();
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("ryMDi5gL-WOpy[=)");
      characterReader0.mark();
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      characterReader0.advance();
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("xm68o5K");
      characterReader0.unconsume();
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("ryMDi5gL-WOpy[=)");
      String string0 = characterReader0.toString();
      assertEquals("ryMDi5gL-WOpy[=)", string0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("ppQ;3vt5!f[bP;-");
      String string0 = characterReader0.consumeAsString();
      assertEquals("p", string0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("cAF~(Ue$Z!Exlq*#");
      String string0 = characterReader0.consumeTo('C');
      assertEquals("cAF~(Ue$Z!Exlq*#", string0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("ryMDi5gL-WOpy[=)");
      int int0 = characterReader0.pos();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("ryMDi5gL-WOpy[=)");
      String string0 = characterReader0.consumeLetterThenDigitSequence();
      assertEquals("ryMDi5", string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("ryMDi5gL-WOpy[=)");
      String string0 = characterReader0.consumeToEnd();
      assertEquals("ryMDi5gL-WOpy[=)", string0);
      
      boolean boolean0 = characterReader0.matchesLetter();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("ppQ;3vt5!f[bP;-");
      char char0 = characterReader0.current();
      assertEquals('p', char0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("ppQ;3vt5!f[bP;-");
      char[] charArray0 = new char[1];
      String string0 = characterReader0.consumeToAny(charArray0);
      assertEquals("ppQ;3vt5!f[bP;-", string0);
      
      char char0 = characterReader0.current();
      assertEquals('\uFFFF', char0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      char char0 = characterReader0.consume();
      assertEquals('\uFFFF', char0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("ryMDi5gL-WOpy[=)");
      boolean boolean0 = characterReader0.containsIgnoreCase("Y 7");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Y 7");
      boolean boolean0 = characterReader0.containsIgnoreCase("Y 7");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      String string0 = characterReader0.consumeTo("Must be true");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("org.jsoup.parser.CharacterReader");
      String string0 = characterReader0.consumeTo("org.jsoup.parser.CharacterReader");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("ppQ;3vt5!f[bP;-");
      char[] charArray0 = new char[9];
      charArray0[0] = '-';
      String string0 = characterReader0.consumeToAny(charArray0);
      assertEquals("ppQ;3vt5!f[bP;", string0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("ryMDi5gL-WOpy[=)");
      boolean boolean0 = characterReader0.matchConsumeIgnoreCase("ryMDi5gL-WOpy[=)");
      assertTrue(boolean0);
      
      char[] charArray0 = new char[1];
      String string0 = characterReader0.consumeToAny(charArray0);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      String string0 = characterReader0.consumeLetterSequence();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("|jI");
      String string0 = characterReader0.consumeLetterSequence();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      String string0 = characterReader0.consumeLetterThenDigitSequence();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("|jI");
      String string0 = characterReader0.consumeLetterThenDigitSequence();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Array must not contain any null objects");
      String string0 = characterReader0.consumeLetterThenDigitSequence();
      assertEquals("Array", string0);
      
      String string1 = characterReader0.consumeDigitSequence();
      assertEquals("", string1);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      String string0 = characterReader0.consumeHexSequence();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Rjc\"^lkc9%Q{ff'd-");
      characterReader0.consume();
      char char0 = characterReader0.consume();
      assertEquals('j', char0);
      
      String string0 = characterReader0.consumeHexSequence();
      assertEquals("c", string0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("1_");
      String string0 = characterReader0.consumeHexSequence();
      assertEquals("1", string0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Array must not contain any null objects");
      String string0 = characterReader0.consumeHexSequence();
      assertEquals("A", string0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      String string0 = characterReader0.consumeDigitSequence();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("6pyb/");
      String string0 = characterReader0.consumeDigitSequence();
      assertEquals("6", string0);
      
      boolean boolean0 = characterReader0.matchConsume("6pyb/");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      boolean boolean0 = characterReader0.matches('@');
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Rjc\"^lkc9%Q{ff'd-");
      boolean boolean0 = characterReader0.matches('^');
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("ppQ;3vt5!f[bP;-");
      String string0 = characterReader0.consumeLetterSequence();
      assertEquals("ppQ", string0);
      
      boolean boolean0 = characterReader0.matches(';');
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("ryMDi5gL-WOpy[=)");
      boolean boolean0 = characterReader0.matchConsume("Y 7");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("1_");
      boolean boolean0 = characterReader0.matchConsume("1_");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("6pyb/");
      boolean boolean0 = characterReader0.matchConsumeIgnoreCase("org.jsoup.parser.CharacterReader");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("ryMDi5gL-WOpy[=)");
      boolean boolean0 = characterReader0.matchConsumeIgnoreCase("Must be true");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("ryMDi5gL-WOpy[=)");
      boolean boolean0 = characterReader0.matchConsumeIgnoreCase("ryMDi5gL-WOpy[=)");
      char[] charArray0 = new char[1];
      boolean boolean1 = characterReader0.matchesAny(charArray0);
      assertFalse(boolean1 == boolean0);
      assertFalse(boolean1);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Rjc\"^lkc9%Q{ff-d-");
      char[] charArray0 = new char[3];
      boolean boolean0 = characterReader0.matchesAny(charArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader(" Ie:FSrO<Z\"q9\\");
      String string0 = characterReader0.consumeTo('r');
      assertEquals(" Ie:FS", string0);
      
      char[] charArray0 = new char[7];
      charArray0[0] = 'r';
      boolean boolean0 = characterReader0.matchesAny(charArray0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("ryMDi5gL-WOpy[=)");
      boolean boolean0 = characterReader0.matchesLetter();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("/-f<2h^QC]nDSvBj.");
      boolean boolean0 = characterReader0.matchesLetter();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Rjc\"^lkc9%Q{ff'd-");
      boolean boolean0 = characterReader0.matchesLetter();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("|jI");
      boolean boolean0 = characterReader0.matchesLetter();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("/-f<2h^QC]nDSvBj.");
      boolean boolean0 = characterReader0.matchesDigit();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      boolean boolean0 = characterReader0.matchesDigit();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("ppQ;3vt5!f[bP;-");
      boolean boolean0 = characterReader0.matchesDigit();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("6.Zk\"#OO1Hf~3");
      boolean boolean0 = characterReader0.matchesDigit();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("1_");
      boolean boolean0 = characterReader0.containsIgnoreCase("1_");
      assertTrue(boolean0);
  }
}
