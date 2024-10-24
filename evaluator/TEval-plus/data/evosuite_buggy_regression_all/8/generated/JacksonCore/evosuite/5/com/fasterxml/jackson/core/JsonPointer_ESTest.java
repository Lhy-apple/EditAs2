/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 02:54:09 GMT 2023
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
      JsonPointer jsonPointer0 = JsonPointer.EMPTY;
      jsonPointer0.hashCode();
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.valueOf("");
      String string0 = jsonPointer0.getMatchingProperty();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("!#");
      JsonPointer jsonPointer1 = jsonPointer0.tail();
      assertTrue(jsonPointer0.mayMatchProperty());
      assertEquals("!#", jsonPointer0.toString());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertTrue(jsonPointer1.matches());
      assertEquals("#", jsonPointer0.getMatchingProperty());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.valueOf("");
      String string0 = jsonPointer0.toString();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      JsonPointer jsonPointer0 = new JsonPointer();
      boolean boolean0 = jsonPointer0.matches();
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.compile("");
      int int0 = jsonPointer0.getMatchingIndex();
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.valueOf((String) null);
      assertEquals((-1), jsonPointer0.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.valueOf("/%Xz&~xe}");
      JsonPointer jsonPointer1 = jsonPointer0.matchProperty("/%Xz&~xe}");
      assertEquals("%Xz&~xe}", jsonPointer0.getMatchingProperty());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertNull(jsonPointer1);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      try { 
        JsonPointer.compile("Invalid input: JSON Pointer expression must start with '/': \"");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid input: JSON Pointer expression must start with '/': \"Invalid input: JSON Pointer expression must start with '/': \"\"
         //
         verifyException("com.fasterxml.jackson.core.JsonPointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("Invalid input: JSON Pointer expression must start with '/': \"");
      boolean boolean0 = jsonPointer0.matches();
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertEquals("nvalid input: JSON Pointer expression must start with '", jsonPointer0.getMatchingProperty());
      assertTrue(jsonPointer0.mayMatchProperty());
      assertEquals("Invalid input: JSON Pointer expression must start with '/': \"", jsonPointer0.toString());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      JsonPointer jsonPointer0 = new JsonPointer();
      boolean boolean0 = jsonPointer0.mayMatchProperty();
      assertEquals((-1), jsonPointer0.getMatchingIndex());
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
      JsonPointer jsonPointer0 = JsonPointer._parseTail("W5");
      boolean boolean0 = jsonPointer0.mayMatchElement();
      assertTrue(jsonPointer0.mayMatchProperty());
      assertEquals("5", jsonPointer0.getMatchingProperty());
      assertEquals("W5", jsonPointer0.toString());
      assertFalse(jsonPointer0.matches());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.valueOf("");
      JsonPointer jsonPointer1 = jsonPointer0.matchProperty("");
      assertNull(jsonPointer1);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail(">v1-");
      JsonPointer jsonPointer1 = new JsonPointer(">v1-", ">v1-", jsonPointer0);
      assertTrue(jsonPointer1.mayMatchProperty());
      
      JsonPointer jsonPointer2 = jsonPointer1.matchProperty(">v1-");
      assertNotNull(jsonPointer2);
      assertEquals(">v1-", jsonPointer2.toString());
      assertEquals("v1-", jsonPointer2.getMatchingProperty());
      assertEquals((-1), jsonPointer2.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      JsonPointer jsonPointer0 = new JsonPointer();
      jsonPointer0.matchElement(6);
      assertEquals((-1), jsonPointer0.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.EMPTY;
      JsonPointer jsonPointer1 = jsonPointer0.matchElement((-1));
      assertNull(jsonPointer1);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      JsonPointer jsonPointer0 = new JsonPointer();
      JsonPointer jsonPointer1 = new JsonPointer();
      boolean boolean0 = jsonPointer0.equals(jsonPointer1);
      assertTrue(boolean0);
      assertEquals((-1), jsonPointer1.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      JsonPointer jsonPointer0 = new JsonPointer();
      boolean boolean0 = jsonPointer0.equals(jsonPointer0);
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.EMPTY;
      boolean boolean0 = jsonPointer0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.EMPTY;
      boolean boolean0 = jsonPointer0.equals(".6");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("I(\"ICyAk/~e2/");
      assertFalse(jsonPointer0.matches());
      assertEquals("(\"ICyAk", jsonPointer0.getMatchingProperty());
      assertFalse(jsonPointer0.mayMatchElement());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      // Undeclared exception!
      try { 
        JsonPointer._parseTail("h8d8[5'6n09");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // For input string: \"8d8[5'6n09\"
         //
         verifyException("java.lang.NumberFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("Z*H.^tFS(Ry7mP~");
      assertFalse(jsonPointer0.matches());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertEquals("Z*H.^tFS(Ry7mP~", jsonPointer0.toString());
      assertEquals("*H.^tFS(Ry7mP~", jsonPointer0.getMatchingProperty());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("n%ZE~,)Q'wTQP,^~");
      assertEquals("%ZE~,)Q'wTQP,^~", jsonPointer0.getMatchingProperty());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertEquals("n%ZE~,)Q'wTQP,^~", jsonPointer0.toString());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseQuotedTail("<BnSm=1eIy0~)!DJID", 10);
      assertEquals("BnSm=1eI~~)!DJID", jsonPointer0.getMatchingProperty());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertFalse(jsonPointer0.matches());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("+=ubwByL2m~1");
      assertEquals("=ubwByL2m/", jsonPointer0.getMatchingProperty());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertEquals("+=ubwByL2m~1", jsonPointer0.toString());
  }
}
