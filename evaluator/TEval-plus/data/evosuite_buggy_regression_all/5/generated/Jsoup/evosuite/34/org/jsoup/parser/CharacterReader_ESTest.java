/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:12:54 GMT 2023
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
      CharacterReader characterReader0 = new CharacterReader("6!");
      characterReader0.rewindToMark();
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("3f^YFhz&c8'iS$?)vP");
      characterReader0.mark();
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("3f^YFhz&c8'iS$?)vP");
      characterReader0.unconsume();
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("os^b^v");
      String string0 = characterReader0.toString();
      assertEquals("os^b^v", string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("E6!N>");
      String string0 = characterReader0.consumeTo('<');
      assertEquals("E6!N>", string0);
      
      char char0 = characterReader0.current();
      assertEquals('\uFFFF', char0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("6!");
      int int0 = characterReader0.pos();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("os^b^v");
      char[] charArray0 = new char[2];
      boolean boolean0 = characterReader0.matchesAny(charArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("-k5w~AaBZsi54kuZ4");
      char char0 = characterReader0.current();
      assertEquals('-', char0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("E6!N>");
      String string0 = characterReader0.consumeTo('<');
      assertEquals("E6!N>", string0);
      
      char char0 = characterReader0.consume();
      assertEquals('\uFFFF', char0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("E6!N>");
      String string0 = characterReader0.consumeTo('E');
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("E6!N>");
      boolean boolean0 = characterReader0.containsIgnoreCase("E6!N>");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("org.jsoup.parser.CharacterReader");
      boolean boolean0 = characterReader0.containsIgnoreCase("org.jsoup.parser.CharacterReader");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("'EhH# l=DH*+K:V81");
      String string0 = characterReader0.consumeTo('.');
      String string1 = characterReader0.consumeTo("1&'|KB=z");
      assertFalse(string1.equals((Object)string0));
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("{Qh ~?H_Dqb3K!`");
      String string0 = characterReader0.consumeTo("{Qh ~?H_Dqb3K!`");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("os^b^v");
      char[] charArray0 = new char[5];
      charArray0[0] = '^';
      String string0 = characterReader0.consumeToAny(charArray0);
      assertEquals("os", string0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("3f^YFhz&c8'iS$?)vP");
      characterReader0.consumeTo('<');
      char[] charArray0 = new char[6];
      String string0 = characterReader0.consumeToAny(charArray0);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("'EhH# l=DH*+K:V81");
      String string0 = characterReader0.consumeTo('.');
      assertEquals("'EhH# l=DH*+K:V81", string0);
      
      String string1 = characterReader0.consumeLetterSequence();
      assertEquals("", string1);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("os^b^v");
      String string0 = characterReader0.consumeLetterSequence();
      assertEquals("os", string0);
      
      char[] charArray0 = new char[2];
      charArray0[0] = '^';
      boolean boolean0 = characterReader0.matchesAny(charArray0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("org.jsoup.parser.CharacterReader");
      String string0 = characterReader0.consumeLetterSequence();
      assertEquals("org", string0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("E6!N>");
      String string0 = characterReader0.consumeLetterSequence();
      assertEquals("E", string0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("|Ov");
      String string0 = characterReader0.consumeLetterSequence();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader(";j)$+hv6PK%#n");
      char char0 = characterReader0.consume();
      assertEquals(';', char0);
      
      String string0 = characterReader0.consumeLetterThenDigitSequence();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("E6!N>");
      String string0 = characterReader0.consumeLetterThenDigitSequence();
      assertEquals("E6", string0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("3f^YFhz&c8'iS$?)vP");
      String string0 = characterReader0.consumeTo('<');
      assertEquals("3f^YFhz&c8'iS$?)vP", string0);
      
      String string1 = characterReader0.consumeHexSequence();
      assertEquals("", string1);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("3f^YFhz&c8'iS$?)vP");
      String string0 = characterReader0.consumeHexSequence();
      assertEquals("3f", string0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("E6!N>");
      String string0 = characterReader0.consumeHexSequence();
      assertEquals("E6", string0);
      
      boolean boolean0 = characterReader0.matchConsumeIgnoreCase("E6");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("WBSxn");
      String string0 = characterReader0.consumeLetterThenDigitSequence();
      assertEquals("WBSxn", string0);
      
      String string1 = characterReader0.consumeDigitSequence();
      assertEquals("", string1);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("os^b^v");
      String string0 = characterReader0.consumeDigitSequence();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("E6!N>");
      String string0 = characterReader0.consumeAsString();
      assertEquals("E", string0);
      
      String string1 = characterReader0.consumeDigitSequence();
      assertEquals("6", string1);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("|Ov");
      char[] charArray0 = new char[5];
      String string0 = characterReader0.consumeToAny(charArray0);
      assertEquals("|Ov", string0);
      
      boolean boolean0 = characterReader0.matches('(');
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("E6!N>");
      boolean boolean0 = characterReader0.matches(']');
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("-k5w~AaBZsi54kuZ4");
      boolean boolean0 = characterReader0.matches('-');
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("maP)*X={ lt4~");
      boolean boolean0 = characterReader0.matches("os^b^v");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("E6!N>");
      boolean boolean0 = characterReader0.matchConsumeIgnoreCase("E6");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("os^b^v");
      boolean boolean0 = characterReader0.matchConsumeIgnoreCase("maP)*X={ lt4~");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      char[] charArray0 = new char[2];
      boolean boolean0 = characterReader0.matchesAny(charArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("E6!N>");
      boolean boolean0 = characterReader0.matchesLetter();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("os^b^v");
      String string0 = characterReader0.consumeToEnd();
      assertEquals("os^b^v", string0);
      
      boolean boolean0 = characterReader0.matchesLetter();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("-k5w~AaBZsi54kuZ4");
      boolean boolean0 = characterReader0.matchesLetter();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("{Qh ~?H_Dqb3K!`");
      boolean boolean0 = characterReader0.matchesLetter();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Array must not contain any null objects");
      String string0 = characterReader0.consumeHexSequence();
      assertEquals("A", string0);
      
      boolean boolean0 = characterReader0.matchesLetter();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("3f^YFhz&c8'iS$?)vP");
      boolean boolean0 = characterReader0.matchesDigit();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("3f^YFhz&c8'iS$?)vP");
      String string0 = characterReader0.consumeTo('<');
      assertEquals("3f^YFhz&c8'iS$?)vP", string0);
      
      boolean boolean0 = characterReader0.matchesDigit();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("-k5w~AaBZsi54kuZ4");
      boolean boolean0 = characterReader0.matchesDigit();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("E6!N>");
      boolean boolean0 = characterReader0.matchesDigit();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("'EhH# l=DH*+K:V81");
      characterReader0.advance();
      boolean boolean0 = characterReader0.matchConsume("'EhH# l=DH*+K:V81");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("63Rv.");
      boolean boolean0 = characterReader0.matchConsume("63Rv.");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("os^b^v");
      boolean boolean0 = characterReader0.containsIgnoreCase("os^b^v");
      assertTrue(boolean0);
  }
}
