/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:04:16 GMT 2023
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
      CharacterReader characterReader0 = new CharacterReader("kx6C,DK@`m");
      characterReader0.rewindToMark();
      assertEquals("kx6C,DK@`m", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("hHgi\"Iq1C%7KK({hGVh");
      boolean boolean0 = characterReader0.matchConsumeIgnoreCase("'K|^/!@cS\n");
      assertEquals("hHgi\"Iq1C%7KK({hGVh", characterReader0.toString());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("{ ?]WvI{b1[r x,T");
      characterReader0.mark();
      assertEquals("{ ?]WvI{b1[r x,T", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("T&XAdSr+/bFA$IK@i#F");
      characterReader0.unconsume();
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("T&XAdSr+/bFA$IK@i#F");
      String string0 = characterReader0.toString();
      assertEquals("T&XAdSr+/bFA$IK@i#F", string0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("yCd?U8,E]H-S&HpwO4");
      characterReader0.consumeAsString();
      assertEquals("yCd?U8,E]H-S&HpwO4", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("T&XAdSr+/bFA$IK@i#F");
      characterReader0.consumeTo('h');
      characterReader0.consumeLetterSequence();
      assertEquals("", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("T&XAdSr+/bFA$IK@i#F");
      int int0 = characterReader0.pos();
      assertEquals("T&XAdSr+/bFA$IK@i#F", characterReader0.toString());
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("org.jsoup.parser.CharacterReader");
      char char0 = characterReader0.current();
      assertEquals('o', char0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("7`N;i");
      assertEquals("7`N;i", characterReader0.toString());
      
      characterReader0.consumeToEnd();
      char char0 = characterReader0.current();
      assertEquals('\uFFFF', char0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("T@zJFR'=1K].n");
      char char0 = characterReader0.consume();
      assertEquals("@zJFR'=1K].n", characterReader0.toString());
      assertEquals('T', char0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("<JD}\";x[\"qX+_U=~xA$");
      characterReader0.consumeTo('|');
      assertEquals("", characterReader0.toString());
      
      char char0 = characterReader0.consume();
      assertEquals('\uFFFF', char0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("hHgi\"Iq1C%7KK({hGVh");
      characterReader0.consumeTo('1');
      characterReader0.consumeHexSequence();
      boolean boolean0 = characterReader0.matchesDigit();
      assertEquals("%7KK({hGVh", characterReader0.toString());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("kx6C,DK@`m");
      characterReader0.consumeTo("qG+}");
      characterReader0.consumeDigitSequence();
      assertEquals("", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("kx6C,DK@`m");
      String string0 = characterReader0.consumeTo("kx6C,DK@`m");
      assertEquals("kx6C,DK@`m", characterReader0.toString());
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("kx6C,DK@`m");
      char[] charArray0 = new char[8];
      characterReader0.consumeToAny(charArray0);
      boolean boolean0 = characterReader0.matchesDigit();
      assertEquals("", characterReader0.toString());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("T&XAdSr+/bFA$IK@i#F");
      String string0 = characterReader0.consumeTo('h');
      String string1 = characterReader0.consumeToAny((char[]) null);
      assertFalse(string1.equals((Object)string0));
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("kx6C,DK@`m");
      characterReader0.consumeLetterSequence();
      characterReader0.consumeDigitSequence();
      assertEquals("C,DK@`m", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("X|xH?n)W/KL8p");
      String string0 = characterReader0.consumeLetterSequence();
      assertEquals("|xH?n)W/KL8p", characterReader0.toString());
      assertEquals("X", string0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("8");
      String string0 = characterReader0.consumeHexSequence();
      assertEquals("8", string0);
      assertEquals("", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("kx6C,DK@`m");
      characterReader0.consumeHexSequence();
      assertEquals("kx6C,DK@`m", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("b(5V4MLe9");
      String string0 = characterReader0.consumeHexSequence();
      assertEquals("(5V4MLe9", characterReader0.toString());
      assertEquals("b", string0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("kx6C,DK@`m");
      characterReader0.consumeDigitSequence();
      assertEquals("kx6C,DK@`m", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader(",");
      characterReader0.consumeDigitSequence();
      assertEquals(",", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("bAF^,Zh?R%");
      assertEquals("bAF^,Zh?R%", characterReader0.toString());
      
      characterReader0.consumeTo('7');
      boolean boolean0 = characterReader0.matches('l');
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("T&XAdSr+/bFA$IK@i#F");
      boolean boolean0 = characterReader0.matches('j');
      assertEquals("T&XAdSr+/bFA$IK@i#F", characterReader0.toString());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("T&XAdSr+/bFA$IK@i#F");
      char[] charArray0 = new char[5];
      charArray0[0] = 'X';
      characterReader0.consumeToAny(charArray0);
      boolean boolean0 = characterReader0.matches('X');
      assertEquals("XAdSr+/bFA$IK@i#F", characterReader0.toString());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("T&XAdSr+/bFA$IK@i#F");
      char[] charArray0 = new char[2];
      boolean boolean0 = characterReader0.matchesAny(charArray0);
      assertFalse(boolean0);
      assertEquals("T&XAdSr+/bFA$IK@i#F", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("kx6C,DK@`m");
      char[] charArray0 = new char[8];
      characterReader0.consumeTo("qG+}");
      boolean boolean0 = characterReader0.matchesAny(charArray0);
      assertEquals("", characterReader0.toString());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      char[] charArray0 = new char[2];
      charArray0[0] = '\'';
      CharacterReader characterReader0 = new CharacterReader("YnrJntG'zJ:CzL}qF");
      characterReader0.consumeLetterSequence();
      boolean boolean0 = characterReader0.matchesAny(charArray0);
      assertEquals("'zJ:CzL}qF", characterReader0.toString());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("kx6C,DK@`m");
      boolean boolean0 = characterReader0.matchesLetter();
      assertTrue(boolean0);
      assertEquals("kx6C,DK@`m", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      boolean boolean0 = characterReader0.matchesLetter();
      assertFalse(boolean0);
      assertEquals("", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader(",");
      boolean boolean0 = characterReader0.matchesLetter();
      assertFalse(boolean0);
      assertEquals(",", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("T&XAdSr+/bFA$IK@i#F");
      boolean boolean0 = characterReader0.matchesLetter();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("yCd?U8,E]H-S&HpwO4");
      boolean boolean0 = characterReader0.matchesLetter();
      assertEquals("yCd?U8,E]H-S&HpwO4", characterReader0.toString());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("kx6C,DK@`m");
      boolean boolean0 = characterReader0.matchesDigit();
      assertFalse(boolean0);
      assertEquals("kx6C,DK@`m", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("(5V");
      characterReader0.advance();
      boolean boolean0 = characterReader0.matchesDigit();
      assertEquals("5V", characterReader0.toString());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("-ag|(j@|f-;-1C");
      boolean boolean0 = characterReader0.matchConsume("$i");
      assertFalse(boolean0);
      assertEquals("-ag|(j@|f-;-1C", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("{ ?]WvI{b1[r x,T");
      boolean boolean0 = characterReader0.matchConsume("");
      assertTrue(boolean0);
      assertEquals("{ ?]WvI{b1[r x,T", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      boolean boolean0 = characterReader0.matchConsumeIgnoreCase("");
      assertEquals("", characterReader0.toString());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("T&XAdSr+/bFA$IK@i#F");
      boolean boolean0 = characterReader0.containsIgnoreCase("");
      assertTrue(boolean0);
      assertEquals("T&XAdSr+/bFA$IK@i#F", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("kx6C,DK@`m");
      boolean boolean0 = characterReader0.containsIgnoreCase("kx6C,DK@`m");
      assertEquals("kx6C,DK@`m", characterReader0.toString());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("T&XAdSr+/bFA$IK@i#F");
      boolean boolean0 = characterReader0.containsIgnoreCase("F");
      assertEquals("T&XAdSr+/bFA$IK@i#F", characterReader0.toString());
      assertTrue(boolean0);
  }
}