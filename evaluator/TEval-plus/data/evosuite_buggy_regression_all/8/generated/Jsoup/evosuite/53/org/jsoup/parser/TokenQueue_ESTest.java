/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 03:32:04 GMT 2023
 */

package org.jsoup.parser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.parser.TokenQueue;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TokenQueue_ESTest extends TokenQueue_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("{xAT");
      tokenQueue0.chompTo("width must be > 0");
      assertEquals('\u0000', tokenQueue0.peek());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("up/_E");
      boolean boolean0 = tokenQueue0.matchesCS("up/_E");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("Tg=DP&Rz");
      Character character0 = Character.valueOf('#');
      tokenQueue0.addFirst(character0);
      assertFalse(tokenQueue0.matchesWord());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("Array must not contain any null objects");
      tokenQueue0.chompToIgnoreCase("  ");
      assertEquals("", tokenQueue0.toString());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("_YQHqhGk\"x%F R");
      String string0 = tokenQueue0.toString();
      assertEquals("_YQHqhGk\"x%F R", string0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("@?>F@Yh/zop)");
      tokenQueue0.chompBalanced('@', '@');
      assertEquals("", tokenQueue0.toString());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("Queue did not match expected sequence");
      char char0 = tokenQueue0.peek();
      assertEquals('Q', char0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("");
      assertTrue(tokenQueue0.isEmpty());
      
      char char0 = tokenQueue0.peek();
      assertEquals('\u0000', char0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("Queue not long enough to consume sequence");
      String[] stringArray0 = new String[1];
      tokenQueue0.consumeAttributeKey();
      stringArray0[0] = "Queue not long enough to consume sequence";
      tokenQueue0.consumeToAny(stringArray0);
      assertEquals("", tokenQueue0.toString());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("?;q=^");
      String[] stringArray0 = new String[1];
      stringArray0[0] = "?;q=^";
      String string0 = tokenQueue0.consumeToAny(stringArray0);
      assertEquals('?', tokenQueue0.peek());
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("[f;y:5M9OXgk(J6S'Q#");
      tokenQueue0.matchChomp("[f;y:5M9OXgk(J6S'Q#");
      assertEquals("", tokenQueue0.toString());
      
      boolean boolean0 = tokenQueue0.matchesAny((char[]) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("L3_itiejUC]b");
      tokenQueue0.consumeAttributeKey();
      assertEquals("]b", tokenQueue0.toString());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("{xAT");
      tokenQueue0.consume("{xAT");
      boolean boolean0 = tokenQueue0.matchesStartTag();
      assertEquals("", tokenQueue0.toString());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("<eze1<i~");
      boolean boolean0 = tokenQueue0.matchesStartTag();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("%@,^y8EPb,?k05_@1R");
      boolean boolean0 = tokenQueue0.matchesStartTag();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("<&&>@");
      boolean boolean0 = tokenQueue0.matchesStartTag();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("   ");
      boolean boolean0 = tokenQueue0.consumeWhitespace();
      assertEquals("", tokenQueue0.toString());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("[F)b");
      boolean boolean0 = tokenQueue0.consumeWhitespace();
      assertFalse(boolean0);
      assertEquals('[', tokenQueue0.peek());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("up/_6KE");
      tokenQueue0.chompToIgnoreCase("up/_6KE");
      tokenQueue0.consumeWord();
      assertEquals("", tokenQueue0.toString());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("");
      tokenQueue0.advance();
      assertTrue(tokenQueue0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("k!!92R");
      tokenQueue0.advance();
      assertEquals('!', tokenQueue0.peek());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("<eze1<i~");
      // Undeclared exception!
      try { 
        tokenQueue0.consume("&drH6");
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Queue did not match expected sequence
         //
         verifyException("org.jsoup.parser.TokenQueue", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("PX!W~!=zm>{Q");
      tokenQueue0.chompTo("PX!W~!=zm>{Q");
      assertEquals("", tokenQueue0.toString());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("PYXp!W~;=zm>{Q");
      tokenQueue0.consumeToIgnoreCase("fxlu)vnot long enough to consume sequence");
      assertEquals("", tokenQueue0.toString());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("up/_6KE");
      tokenQueue0.chompBalanced(':', 'u');
      assertEquals("p/_6KE", tokenQueue0.toString());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      String string0 = TokenQueue.unescape("?\\[,<");
      assertEquals("?[,<", string0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("up/_6KE");
      tokenQueue0.consumeWord();
      assertEquals("/_6KE", tokenQueue0.toString());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("");
      String string0 = tokenQueue0.consumeTagName();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("L3_itiejUC]b");
      tokenQueue0.consumeTagName();
      assertEquals("]b", tokenQueue0.toString());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("0xA");
      tokenQueue0.consumeElementSelector();
      assertEquals("", tokenQueue0.toString());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("hyi2-)ew&s#j(aa");
      tokenQueue0.consumeElementSelector();
      assertEquals(')', tokenQueue0.peek());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("");
      String string0 = tokenQueue0.consumeCssIdentifier();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("_I<)jAb)Sl:iM%a");
      tokenQueue0.consumeCssIdentifier();
      assertEquals("<)jAb)Sl:iM%a", tokenQueue0.toString());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("");
      String string0 = tokenQueue0.consumeAttributeKey();
      assertEquals("", string0);
  }
}