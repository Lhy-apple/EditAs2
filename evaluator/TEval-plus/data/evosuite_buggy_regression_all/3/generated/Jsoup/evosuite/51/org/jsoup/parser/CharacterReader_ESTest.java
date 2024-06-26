/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:51:52 GMT 2023
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
      CharacterReader characterReader0 = new CharacterReader(";x ..Dcm={g~zL");
      characterReader0.mark();
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("String must not be empty");
      characterReader0.advance();
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("`^']!hz2-x}`^]H");
      characterReader0.unconsume();
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("String ist nt `e 6mpty");
      String string0 = characterReader0.toString();
      assertEquals("String ist nt `e 6mpty", string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader(";x ..Dcm={g~zL");
      String string0 = characterReader0.consumeAsString();
      assertEquals(";", string0);
      
      boolean boolean0 = characterReader0.matchesIgnoreCase(";");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("`^']!hzp\"-i}`^]H");
      String string0 = characterReader0.consumeTo("-~+`wNw@, ");
      assertEquals("`^']!hzp\"-i}`^]H", string0);
      
      String string1 = characterReader0.consumeTagName();
      assertEquals("", string1);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Cg=");
      int int0 = characterReader0.pos();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("6[J^iyS$DY,wO");
      String string0 = characterReader0.consumeLetterThenDigitSequence();
      assertEquals("6", string0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("`^']!hz2-i}`^]H");
      String string0 = characterReader0.consumeTagName();
      assertEquals("`^']!hz2-i}`^]H", string0);
      
      boolean boolean0 = characterReader0.matchesLetter();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("7fPL1L79#w");
      char char0 = characterReader0.current();
      assertEquals('7', char0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      char char0 = characterReader0.current();
      assertEquals('\uFFFF', char0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("!?] \"\\");
      char char0 = characterReader0.consume();
      assertEquals('!', char0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      char char0 = characterReader0.consume();
      assertEquals('\uFFFF', char0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Jk_Cs&AQ");
      String string0 = characterReader0.consumeTo('2');
      assertEquals("Jk_Cs&AQ", string0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("`^']!hz\"-i}`^]H");
      String string0 = characterReader0.consumeTo('`');
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("5!z i");
      boolean boolean0 = characterReader0.containsIgnoreCase("5!z i");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("6!!Q *_");
      boolean boolean0 = characterReader0.containsIgnoreCase("6!!Q *_");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("String must not be empty");
      String string0 = characterReader0.consumeTo("String must not be empty");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Array mustenot coGtain aAy n,l objects");
      char[] charArray0 = new char[1];
      charArray0[0] = 'y';
      String string0 = characterReader0.consumeToAny(charArray0);
      assertEquals("Arra", string0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("`^']!hz2-x}`^]H");
      String string0 = characterReader0.consumeData();
      assertEquals("`^']!hz2-x}`^]H", string0);
      
      char[] charArray0 = new char[7];
      String string1 = characterReader0.consumeToAny(charArray0);
      assertEquals("", string1);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      char[] charArray0 = new char[0];
      String string0 = characterReader0.consumeToAnySorted(charArray0);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Y(rpx.frw [-C?j~i-");
      char[] charArray0 = new char[2];
      String string0 = characterReader0.consumeToAnySorted(charArray0);
      assertEquals("Y(rpx.frw [-C?j~i-", string0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Y(rpx.frw [-C?j~i-");
      char[] charArray0 = new char[2];
      charArray0[1] = ' ';
      String string0 = characterReader0.consumeToAnySorted(charArray0);
      assertEquals("Y(rpx.frw", string0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("&a");
      String string0 = characterReader0.consumeData();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("[:_FK?;<");
      String string0 = characterReader0.consumeData();
      assertEquals("[:_FK?;", string0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("5!z i");
      String string0 = characterReader0.consumeTagName();
      assertEquals("5!z", string0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("3SSS}L>@m!9{W");
      String string0 = characterReader0.consumeTagName();
      assertEquals("3SSS}L", string0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Nl");
      String string0 = characterReader0.consumeLetterSequence();
      assertEquals("Nl", string0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("!?] \"\\");
      String string0 = characterReader0.consumeLetterSequence();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("OH/Y{cg]NwjB1");
      String string0 = characterReader0.consumeLetterSequence();
      assertEquals("OH", string0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Fa");
      String string0 = characterReader0.consumeLetterThenDigitSequence();
      assertEquals("Fa", string0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("cG{8d'a Ep-9");
      String string0 = characterReader0.consumeLetterThenDigitSequence();
      assertEquals("cG", string0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("#F\"5r{i`HQ$!E");
      String string0 = characterReader0.consumeLetterThenDigitSequence();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("a");
      String string0 = characterReader0.consumeHexSequence();
      assertEquals("a", string0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("6!! 5_3\"l");
      String string0 = characterReader0.consumeHexSequence();
      assertEquals("6", string0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("CvG\"1_3YE");
      String string0 = characterReader0.consumeHexSequence();
      assertEquals("C", string0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      String string0 = characterReader0.consumeDigitSequence();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("2(M!z iVO");
      String string0 = characterReader0.consumeDigitSequence();
      assertEquals("2", string0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("String must not be empty");
      String string0 = characterReader0.consumeDigitSequence();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("`^']!hz2-i}`^]H");
      String string0 = characterReader0.consumeTagName();
      assertEquals("`^']!hz2-i}`^]H", string0);
      
      boolean boolean0 = characterReader0.matches('+');
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("7fPL1L79#w");
      boolean boolean0 = characterReader0.matches('7');
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Array must not contain any null objects");
      boolean boolean0 = characterReader0.matches('q');
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("?/T");
      boolean boolean0 = characterReader0.matches("?");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("`^']!hz2-i}`^]H");
      String string0 = characterReader0.consumeTagName();
      assertEquals("`^']!hz2-i}`^]H", string0);
      
      boolean boolean0 = characterReader0.matchConsume("`^']!hz2-i}`^]H");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("?/T");
      String string0 = characterReader0.consumeTagName();
      assertEquals("?", string0);
      
      boolean boolean0 = characterReader0.matches("?");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Array must not contain any null objectj");
      boolean boolean0 = characterReader0.matchConsumeIgnoreCase("Array must not contain any null objectj");
      boolean boolean1 = characterReader0.matchConsumeIgnoreCase("Array must not contain any null objectj");
      assertFalse(boolean1 == boolean0);
      assertFalse(boolean1);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("7fPL1L79#w");
      char[] charArray0 = new char[3];
      boolean boolean0 = characterReader0.matchesAny(charArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      char[] charArray0 = new char[1];
      CharacterReader characterReader0 = new CharacterReader("");
      boolean boolean0 = characterReader0.matchesAny(charArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("7fPL1L79#w");
      char[] charArray0 = new char[3];
      charArray0[0] = '7';
      boolean boolean0 = characterReader0.matchesAny(charArray0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("String must not be empty");
      String string0 = characterReader0.consumeData();
      assertEquals("String must not be empty", string0);
      
      boolean boolean0 = characterReader0.matchesAnySorted((char[]) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("7fPL1L79#w");
      char[] charArray0 = new char[3];
      boolean boolean0 = characterReader0.matchesAnySorted(charArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("7fPL1L79#w");
      char[] charArray0 = new char[3];
      charArray0[1] = '7';
      boolean boolean0 = characterReader0.matchesAnySorted(charArray0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("|%@");
      boolean boolean0 = characterReader0.matchesLetter();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("7EkvF`v=o}gc`");
      boolean boolean0 = characterReader0.matchesLetter();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("String iust not be empty");
      boolean boolean0 = characterReader0.matchesLetter();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("org.jsoup.parser.CharacterReader");
      boolean boolean0 = characterReader0.matchesLetter();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("!!Q *_");
      boolean boolean0 = characterReader0.matchesDigit();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      boolean boolean0 = characterReader0.matchesDigit();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("6!!Q *_");
      boolean boolean0 = characterReader0.matchesDigit();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("org.jsoup.parser.CharacterReader");
      boolean boolean0 = characterReader0.matchesDigit();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      boolean boolean0 = characterReader0.matchConsume("");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("7fPL1L79#w");
      boolean boolean0 = characterReader0.containsIgnoreCase("7fPL1L79#w");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("U,!o}8!ef~");
      String string0 = characterReader0.consumeTagName();
      assertEquals("U,!o}8!ef~", string0);
      
      characterReader0.rewindToMark();
      String string1 = characterReader0.consumeData();
      assertEquals("U,!o}8!ef~", string1);
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("String must not be empty");
      boolean boolean0 = characterReader0.rangeEquals(1, 1, "S");
      assertFalse(boolean0);
  }
}
