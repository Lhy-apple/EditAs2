/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:17:50 GMT 2023
 */

package com.fasterxml.jackson.core;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonPointer;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class JsonPointer_ESTest extends JsonPointer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseQuotedTail("Invalid input: JSON Pointer expression must start with '/': \"", 0);
      jsonPointer0.hashCode();
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertTrue(jsonPointer0.mayMatchProperty());
      assertEquals("~Invalid input: JSON Pointer expression must start with '", jsonPointer0.getMatchingProperty());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("?.O5hhh5<");
      String string0 = jsonPointer0.getMatchingProperty();
      assertEquals("?.O5hhh5<", jsonPointer0.toString());
      assertEquals(".O5hhh5<", string0);
      assertFalse(jsonPointer0.matches());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      JsonPointer jsonPointer0 = new JsonPointer();
      JsonPointer jsonPointer1 = jsonPointer0.tail();
      assertNull(jsonPointer1);
      assertEquals((-1), jsonPointer0.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      JsonPointer jsonPointer0 = new JsonPointer();
      jsonPointer0.toString();
      assertEquals((-1), jsonPointer0.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.valueOf("");
      assertEquals("", jsonPointer0.toString());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      JsonPointer jsonPointer0 = new JsonPointer();
      int int0 = jsonPointer0.getMatchingIndex();
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.compile("/a%xX[u7tB{r");
      assertEquals("/a%xX[u7tB{r", jsonPointer0.toString());
      assertFalse(jsonPointer0.matches());
      assertEquals("a%xX[u7tB{r", jsonPointer0.getMatchingProperty());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertTrue(jsonPointer0.mayMatchProperty());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      // Undeclared exception!
      try { 
        JsonPointer.valueOf("$}4l2");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid input: JSON Pointer expression must start with '/': \"$}4l2\"
         //
         verifyException("com.fasterxml.jackson.core.JsonPointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("\" can not be represented as BigDecimal");
      boolean boolean0 = jsonPointer0.matches();
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertTrue(jsonPointer0.mayMatchProperty());
      assertFalse(boolean0);
      assertEquals(" can not be represented as BigDecimal", jsonPointer0.getMatchingProperty());
      assertEquals("\" can not be represented as BigDecimal", jsonPointer0.toString());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      JsonPointer jsonPointer0 = new JsonPointer();
      boolean boolean0 = jsonPointer0.matches();
      assertTrue(boolean0);
      assertEquals((-1), jsonPointer0.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.EMPTY;
      boolean boolean0 = jsonPointer0.mayMatchProperty();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      JsonPointer jsonPointer0 = new JsonPointer();
      boolean boolean0 = jsonPointer0.mayMatchElement();
      assertFalse(boolean0);
      assertEquals((-1), jsonPointer0.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("\"7");
      boolean boolean0 = jsonPointer0.mayMatchElement();
      assertEquals("\"7", jsonPointer0.toString());
      assertEquals("7", jsonPointer0.getMatchingProperty());
      assertTrue(jsonPointer0.mayMatchProperty());
      assertFalse(jsonPointer0.matches());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      JsonPointer jsonPointer0 = new JsonPointer();
      jsonPointer0.matchProperty("..O5hhh5<");
      assertEquals((-1), jsonPointer0.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      JsonPointer jsonPointer0 = new JsonPointer();
      JsonPointer jsonPointer1 = new JsonPointer("@&", "@&", jsonPointer0);
      assertTrue(jsonPointer1.mayMatchProperty());
      
      JsonPointer jsonPointer2 = jsonPointer1.matchProperty("@&");
      assertSame(jsonPointer2, jsonPointer0);
      assertEquals((-1), jsonPointer2.getMatchingIndex());
      assertNotNull(jsonPointer2);
      assertEquals((-1), jsonPointer1.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("Invalid input: JSON Pointer expression must start with '/': \"");
      JsonPointer jsonPointer1 = jsonPointer0.matchProperty("Invalid input: JSON Pointer expression must start with '/': \"");
      assertEquals("Invalid input: JSON Pointer expression must start with '/': \"", jsonPointer0.toString());
      assertEquals("nvalid input: JSON Pointer expression must start with '", jsonPointer0.getMatchingProperty());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertNull(jsonPointer1);
      assertFalse(jsonPointer0.matches());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.EMPTY;
      JsonPointer jsonPointer1 = jsonPointer0.matchElement((-1126));
      assertNull(jsonPointer1);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("|3");
      JsonPointer jsonPointer1 = jsonPointer0.matchElement(3);
      assertNotNull(jsonPointer1);
      assertEquals("|3", jsonPointer0.toString());
      assertEquals("3", jsonPointer0.getMatchingProperty());
      assertEquals((-1), jsonPointer1.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.EMPTY;
      JsonPointer jsonPointer1 = jsonPointer0.matchElement((-1));
      assertNull(jsonPointer1);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("B,Y}X)s/ejb>%");
      Object object0 = new Object();
      boolean boolean0 = jsonPointer0.equals(object0);
      assertEquals("B,Y}X)s/ejb>%", jsonPointer0.toString());
      assertEquals(",Y}X)s", jsonPointer0.getMatchingProperty());
      assertFalse(jsonPointer0.mayMatchElement());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("B,Y}X)s/ejb>%");
      boolean boolean0 = jsonPointer0.equals(jsonPointer0);
      assertFalse(jsonPointer0.matches());
      assertEquals(",Y}X)s", jsonPointer0.getMatchingProperty());
      assertTrue(boolean0);
      assertEquals("B,Y}X)s/ejb>%", jsonPointer0.toString());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.compile((String) null);
      boolean boolean0 = jsonPointer0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("B,Y}X)s/ejb>%");
      JsonPointer jsonPointer1 = new JsonPointer();
      boolean boolean0 = jsonPointer0.equals(jsonPointer1);
      assertEquals(",Y}X)s", jsonPointer0.getMatchingProperty());
      assertFalse(jsonPointer1.equals((Object)jsonPointer0));
      assertFalse(jsonPointer0.matches());
      assertFalse(jsonPointer0.mayMatchElement());
      assertEquals((-1), jsonPointer1.getMatchingIndex());
      assertEquals("B,Y}X)s/ejb>%", jsonPointer0.toString());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("\"");
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertEquals("", jsonPointer0.getMatchingProperty());
      assertEquals("\"", jsonPointer0.toString());
      assertFalse(jsonPointer0.matches());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      // Undeclared exception!
      try { 
        JsonPointer._parseTail("}1Q0w3h6&7G/(F8UYD");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // For input string: \"1Q0w3h6&7G\"
         //
         verifyException("java.lang.NumberFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("k-/{t~$v/MJ!^#");
      assertEquals("-", jsonPointer0.getMatchingProperty());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("~~");
      assertFalse(jsonPointer0.matches());
      assertEquals("~~", jsonPointer0.toString());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertEquals("~", jsonPointer0.getMatchingProperty());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("*sJ~b]uXf.feam\"~R-");
      assertEquals("sJ~b]uXf.feam\"~R-", jsonPointer0.getMatchingProperty());
      assertFalse(jsonPointer0.mayMatchElement());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseQuotedTail("~~", 0);
      assertFalse(jsonPointer0.matches());
      assertEquals("~~~", jsonPointer0.getMatchingProperty());
      assertEquals("~~", jsonPointer0.toString());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseQuotedTail("0cNq,rRy}3buD&z", 0);
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertFalse(jsonPointer0.matches());
      assertEquals("~cNq,rRy}3buD&z", jsonPointer0.getMatchingProperty());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("b_U[~1CMyZ6}");
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertEquals("_U[/CMyZ6}", jsonPointer0.getMatchingProperty());
      assertFalse(jsonPointer0.matches());
  }
}
