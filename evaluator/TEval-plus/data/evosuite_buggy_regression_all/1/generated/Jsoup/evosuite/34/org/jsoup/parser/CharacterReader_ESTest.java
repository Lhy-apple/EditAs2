/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:47:00 GMT 2023
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
      CharacterReader characterReader0 = new CharacterReader("Array must not cHntai: any null objects");
      characterReader0.rewindToMark();
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Hk0IQ%h*yleSt");
      characterReader0.mark();
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("N9V");
      characterReader0.advance();
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Hk0IQ%h*yleSt");
      String string0 = characterReader0.toString();
      assertEquals("Hk0IQ%h*yleSt", string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Array must not cHntai: any null objects");
      String string0 = characterReader0.consumeAsString();
      assertEquals("A", string0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Hk0IQ%h*yleSt");
      int int0 = characterReader0.pos();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Hk0IQ%h*yleSt");
      boolean boolean0 = characterReader0.matchConsume("Hk0IQ%h*yleSt");
      assertTrue(boolean0);
      
      String string0 = characterReader0.consumeLetterThenDigitSequence();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Array must not contain any null objects");
      char char0 = characterReader0.current();
      assertEquals('A', char0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Hk0IQ%h*yleSt");
      boolean boolean0 = characterReader0.matchConsume("Hk0IQ%h*yleSt");
      assertTrue(boolean0);
      
      char char0 = characterReader0.current();
      assertEquals('\uFFFF', char0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      char char0 = characterReader0.consume();
      assertEquals('\uFFFF', char0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Hk0IQ%h*yleSt");
      String string0 = characterReader0.consumeTo('C');
      assertEquals("Hk0IQ%h*yleSt", string0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("ye;zG@0q");
      String string0 = characterReader0.consumeTo('y');
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("cqqZh5#J9dIr");
      boolean boolean0 = characterReader0.containsIgnoreCase("z");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("ye;zG@0q");
      boolean boolean0 = characterReader0.containsIgnoreCase("ye;zG@0q");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("org.jsoup.helper.Validate");
      String string0 = characterReader0.consumeTo("Object must not be null");
      assertEquals("org.jsoup.helper.Validate", string0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Array must not cHntai: any null objects");
      String string0 = characterReader0.consumeTo("Array must not cHntai: any null objects");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Array must not cHntai: any null objects");
      char[] charArray0 = new char[8];
      charArray0[1] = 'y';
      String string0 = characterReader0.consumeToAny(charArray0);
      assertEquals("Arra", string0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("&prd;b5<1n#");
      String string0 = characterReader0.consumeToEnd();
      assertEquals("&prd;b5<1n#", string0);
      
      char[] charArray0 = new char[1];
      String string1 = characterReader0.consumeToAny(charArray0);
      assertEquals("", string1);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("&prd;b5<1n#");
      String string0 = characterReader0.consumeToEnd();
      assertEquals("&prd;b5<1n#", string0);
      
      String string1 = characterReader0.consumeLetterSequence();
      assertEquals("", string1);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Hk0IQ%h*yleSt");
      String string0 = characterReader0.consumeLetterSequence();
      assertEquals("Hk", string0);
      
      boolean boolean0 = characterReader0.matchesDigit();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("|/.Xb]3I.@");
      String string0 = characterReader0.consumeLetterSequence();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Hk0IQ%h*yleSt");
      String string0 = characterReader0.consumeLetterThenDigitSequence();
      assertEquals("Hk0", string0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("|[:bl5V4Vy,uL0pL");
      String string0 = characterReader0.consumeLetterThenDigitSequence();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Array must not cHntai: any null objects");
      String string0 = characterReader0.consumeLetterThenDigitSequence();
      assertEquals("Array", string0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("%Q$b@H+:I");
      String string0 = characterReader0.consumeToEnd();
      assertEquals("%Q$b@H+:I", string0);
      
      String string1 = characterReader0.consumeHexSequence();
      assertEquals("", string1);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("&*Vv");
      String string0 = characterReader0.consumeHexSequence();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("9}Wq_~gojB.%+z");
      String string0 = characterReader0.consumeHexSequence();
      assertEquals("9", string0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Array must not cHntai: any null objects");
      String string0 = characterReader0.consumeHexSequence();
      assertEquals("A", string0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Hk0IQ%h*yleSt");
      boolean boolean0 = characterReader0.matchConsume("Hk0IQ%h*yleSt");
      assertTrue(boolean0);
      
      characterReader0.unconsume();
      characterReader0.unconsume();
      characterReader0.unconsume();
      String string0 = characterReader0.consumeHexSequence();
      assertEquals("e", string0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("N9V");
      String string0 = characterReader0.consumeToEnd();
      assertEquals("N9V", string0);
      
      String string1 = characterReader0.consumeDigitSequence();
      assertEquals("", string1);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("N9V");
      String string0 = characterReader0.consumeDigitSequence();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("m5,[<<xhp\"cn/K<h");
      char char0 = characterReader0.consume();
      assertEquals('m', char0);
      
      String string0 = characterReader0.consumeDigitSequence();
      assertEquals("5", string0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      boolean boolean0 = characterReader0.matches('7');
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Hk0IQ%h*yleSt");
      boolean boolean0 = characterReader0.matches('a');
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("N9V");
      boolean boolean0 = characterReader0.matches('N');
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Hk0IQ%h*yleSt");
      boolean boolean0 = characterReader0.matchConsume("Hk0IQ%h*yleSt");
      boolean boolean1 = characterReader0.matchConsume("]MdQl-br?KTre");
      assertFalse(boolean1 == boolean0);
      assertFalse(boolean1);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Array)ust nob=cHntai: any null objects");
      boolean boolean0 = characterReader0.matchConsume("#y\"BWV'@kY");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Hk0IQ%h*yleSt");
      boolean boolean0 = characterReader0.matchConsume("Hk0IQ%h*yleSt");
      assertTrue(boolean0);
      
      boolean boolean1 = characterReader0.matchConsumeIgnoreCase("Hk0IQ%h*yleSt");
      assertFalse(boolean1);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("9}Wq_~gojB.%+z");
      boolean boolean0 = characterReader0.matchConsumeIgnoreCase("5hY}a!?av");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Array must not cHntai: any null objects");
      boolean boolean0 = characterReader0.matchConsumeIgnoreCase("Array must not cHntai: any null objects");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Hk0IQ%h*yleSt");
      char[] charArray0 = new char[2];
      boolean boolean0 = characterReader0.matchesAny(charArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("#j");
      char[] charArray0 = new char[1];
      String string0 = characterReader0.consumeToAny(charArray0);
      assertEquals("#j", string0);
      
      boolean boolean0 = characterReader0.matchesAny(charArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("|[:bl5V4Vy,uL0pL");
      char[] charArray0 = new char[7];
      charArray0[0] = '|';
      boolean boolean0 = characterReader0.matchesAny(charArray0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Hk0IQ%h*yleSt");
      boolean boolean0 = characterReader0.matchesLetter();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("&prd;b5<1n#");
      String string0 = characterReader0.consumeToEnd();
      assertEquals("&prd;b5<1n#", string0);
      
      boolean boolean0 = characterReader0.matchesLetter();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("&*Vv");
      boolean boolean0 = characterReader0.matchesLetter();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Hk0IQ%h*yleSt");
      boolean boolean0 = characterReader0.matchConsume("Hk0IQ%h*yleSt");
      assertTrue(boolean0);
      
      characterReader0.unconsume();
      characterReader0.unconsume();
      characterReader0.unconsume();
      boolean boolean1 = characterReader0.matchesLetter();
      assertTrue(boolean1);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("|/.Xb]3I.@");
      boolean boolean0 = characterReader0.matchesLetter();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Hk0IQ%h*yleSt");
      boolean boolean0 = characterReader0.matchesDigit();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Array must not cHntai: any null objects");
      String string0 = characterReader0.consumeToEnd();
      assertEquals("Array must not cHntai: any null objects", string0);
      
      boolean boolean0 = characterReader0.matchesDigit();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("%Q$b@H+:I");
      boolean boolean0 = characterReader0.matchesDigit();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("u%*7b(~m^~0_$`@");
      boolean boolean0 = characterReader0.containsIgnoreCase("u%*7b(~m^~0_$`@");
      assertTrue(boolean0);
  }
}
