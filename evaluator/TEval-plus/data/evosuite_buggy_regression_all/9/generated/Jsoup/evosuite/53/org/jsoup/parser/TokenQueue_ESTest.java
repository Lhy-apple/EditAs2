/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:10:43 GMT 2023
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
      TokenQueue tokenQueue0 = new TokenQueue("XMx'~");
      tokenQueue0.chompTo("XMx'~");
      assertTrue(tokenQueue0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("     onsdf");
      boolean boolean0 = tokenQueue0.matchesCS("     onsdf");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("Array must not contain any null objects");
      tokenQueue0.chompToIgnoreCase("          ");
      assertEquals("", tokenQueue0.toString());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("O axzuR{Q");
      String string0 = tokenQueue0.toString();
      assertEquals("O axzuR{Q", string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue(":jonsdf");
      Character character0 = Character.valueOf('<');
      tokenQueue0.addFirst(character0);
      boolean boolean0 = tokenQueue0.matchesStartTag();
      assertEquals('<', tokenQueue0.peek());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("(b\u0004#p?xc7{hb33(b");
      tokenQueue0.chompBalanced('(', '7');
      assertEquals('{', tokenQueue0.peek());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("1P6`]q");
      tokenQueue0.consumeTo("A_(#IZf@J0q$s6}{Q");
      assertEquals("", tokenQueue0.toString());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("1P6V]q");
      char char0 = tokenQueue0.peek();
      assertEquals('1', char0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("");
      assertTrue(tokenQueue0.isEmpty());
      
      char char0 = tokenQueue0.peek();
      assertEquals('\u0000', char0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("H=R->{J");
      String[] stringArray0 = new String[0];
      tokenQueue0.consumeToAny(stringArray0);
      assertEquals("", tokenQueue0.toString());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("OxaxzzR{Q");
      tokenQueue0.consumeWord();
      String[] stringArray0 = new String[3];
      stringArray0[0] = "OxaxzzR{Q";
      // Undeclared exception!
      try { 
        tokenQueue0.consumeToAny(stringArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("OxaxzzR{Q");
      String[] stringArray0 = new String[3];
      stringArray0[0] = "OxaxzzR{Q";
      tokenQueue0.consumeToAny(stringArray0);
      assertEquals('O', tokenQueue0.peek());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("_nv~1H&O");
      tokenQueue0.consumeElementSelector();
      // Undeclared exception!
      try { 
        tokenQueue0.consume("_nv~1H&O");
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Queue did not match expected sequence
         //
         verifyException("org.jsoup.parser.TokenQueue", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("");
      char[] charArray0 = new char[2];
      boolean boolean0 = tokenQueue0.matchesAny(charArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("");
      boolean boolean0 = tokenQueue0.matchesStartTag();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("x<E");
      boolean boolean0 = tokenQueue0.matchesStartTag();
      assertEquals('x', tokenQueue0.peek());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("x<E");
      tokenQueue0.consumeTagName();
      boolean boolean0 = tokenQueue0.matchesStartTag();
      assertEquals("<E", tokenQueue0.toString());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("");
      boolean boolean0 = tokenQueue0.consumeWhitespace();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("     onsdf");
      boolean boolean0 = tokenQueue0.consumeWhitespace();
      assertEquals("onsdf", tokenQueue0.toString());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("");
      boolean boolean0 = tokenQueue0.matchesWord();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("");
      tokenQueue0.advance();
      assertTrue(tokenQueue0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("OxaxzzR{Q");
      tokenQueue0.advance();
      assertEquals('x', tokenQueue0.peek());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("_nv~1H&O");
      tokenQueue0.consume("_nv~1H&O");
      assertEquals("", tokenQueue0.toString());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("1p6 ]q");
      tokenQueue0.consumeToIgnoreCase("]q");
      assertEquals(']', tokenQueue0.peek());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("k+Sw7:As1ul");
      tokenQueue0.consume();
      tokenQueue0.consumeToIgnoreCase("k+Sw7:As1ul");
      assertEquals("", tokenQueue0.toString());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("(b\u0004#p?xc7{hb33(b");
      tokenQueue0.chompBalanced('(', '(');
      assertEquals("", tokenQueue0.toString());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("B|Aw2l6(2qt");
      tokenQueue0.chompBalanced('P', 'P');
      assertEquals("|Aw2l6(2qt", tokenQueue0.toString());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      String string0 = TokenQueue.unescape("4;\\Kb:kX% ");
      assertEquals("4;Kb:kX% ", string0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("ch:JoNSdf");
      tokenQueue0.consumeTagName();
      assertEquals("", tokenQueue0.toString());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("c>E-HH<m#");
      tokenQueue0.chompToIgnoreCase("c>E-HH<m#");
      tokenQueue0.consumeElementSelector();
      assertEquals("", tokenQueue0.toString());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("GtTmN");
      tokenQueue0.consumeCssIdentifier();
      assertEquals("", tokenQueue0.toString());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("cE-<m<#");
      tokenQueue0.consumeCssIdentifier();
      assertEquals("<m<#", tokenQueue0.toString());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("GtTmN");
      tokenQueue0.consumeAttributeKey();
      assertEquals("", tokenQueue0.toString());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      TokenQueue tokenQueue0 = new TokenQueue("Z_Ac_(i,#NGdzT`ov");
      tokenQueue0.consumeAttributeKey();
      assertEquals("(i,#NGdzT`ov", tokenQueue0.toString());
  }
}
