/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:33:47 GMT 2023
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
      JsonPointer jsonPointer0 = new JsonPointer();
      jsonPointer0.hashCode();
      assertEquals((-1), jsonPointer0.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("[8Wv!.*XmE3w");
      String string0 = jsonPointer0.getMatchingProperty();
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertEquals("8Wv!.*XmE3w", string0);
      assertEquals("[8Wv!.*XmE3w", jsonPointer0.toString());
      assertNotNull(string0);
      assertFalse(jsonPointer0.matches());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.valueOf("");
      JsonPointer jsonPointer1 = jsonPointer0.tail();
      assertNull(jsonPointer1);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("0FJ5SARz@ax-8");
      String string0 = jsonPointer0.toString();
      assertNotNull(string0);
      assertFalse(jsonPointer0.matches());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertTrue(jsonPointer0.mayMatchProperty());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      JsonPointer jsonPointer0 = new JsonPointer();
      int int0 = jsonPointer0.getMatchingIndex();
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.compile((String) null);
      assertEquals((-1), jsonPointer0.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.valueOf("/Qk");
      assertTrue(jsonPointer0.mayMatchProperty());
      assertEquals("Qk", jsonPointer0.getMatchingProperty());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertEquals("/Qk", jsonPointer0.toString());
      assertFalse(jsonPointer0.matches());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      try { 
        JsonPointer.compile("4f%n276F5yxCw]$~d");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid input: JSON Pointer expression must start with '/': \"4f%n276F5yxCw]$~d\"
         //
         verifyException("com.fasterxml.jackson.core.JsonPointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("[h");
      boolean boolean0 = jsonPointer0.matches();
      assertFalse(boolean0);
      assertEquals("h", jsonPointer0.getMatchingProperty());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertTrue(jsonPointer0.mayMatchProperty());
      assertEquals("[h", jsonPointer0.toString());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.EMPTY;
      boolean boolean0 = jsonPointer0.matches();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.EMPTY;
      boolean boolean0 = jsonPointer0.mayMatchProperty();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("[8Wv!.*XmE3w");
      boolean boolean0 = jsonPointer0.mayMatchElement();
      assertFalse(jsonPointer0.matches());
      assertFalse(boolean0);
      assertEquals("8Wv!.*XmE3w", jsonPointer0.getMatchingProperty());
      assertEquals("[8Wv!.*XmE3w", jsonPointer0.toString());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertTrue(jsonPointer0.mayMatchProperty());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("Z9");
      boolean boolean0 = jsonPointer0.mayMatchElement();
      assertEquals("Z9", jsonPointer0.toString());
      assertTrue(jsonPointer0.mayMatchProperty());
      assertFalse(jsonPointer0.matches());
      assertTrue(boolean0);
      assertEquals("9", jsonPointer0.getMatchingProperty());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.EMPTY;
      JsonPointer jsonPointer1 = jsonPointer0.matchProperty("com.fasterxml.jackson.core.JsonPointer");
      assertNull(jsonPointer1);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("H=C");
      assertTrue(jsonPointer0.mayMatchProperty());
      
      JsonPointer jsonPointer1 = jsonPointer0.matchProperty("H=C");
      assertFalse(jsonPointer0.matches());
      assertFalse(jsonPointer0.mayMatchElement());
      assertEquals("H=C", jsonPointer0.toString());
      assertNull(jsonPointer1);
      assertEquals("=C", jsonPointer0.getMatchingProperty());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      JsonPointer jsonPointer0 = new JsonPointer();
      JsonPointer jsonPointer1 = new JsonPointer("\"", "\"", jsonPointer0);
      JsonPointer jsonPointer2 = jsonPointer1.matchProperty("\"");
      assertNotNull(jsonPointer2);
      assertSame(jsonPointer2, jsonPointer0);
      assertEquals((-1), jsonPointer1.getMatchingIndex());
      assertEquals((-1), jsonPointer2.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.EMPTY;
      JsonPointer jsonPointer1 = jsonPointer0.matchElement((-350));
      assertNull(jsonPointer1);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      JsonPointer jsonPointer0 = new JsonPointer();
      jsonPointer0.matchElement((-1));
      assertEquals((-1), jsonPointer0.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail(":");
      JsonPointer jsonPointer1 = new JsonPointer(":", ":", jsonPointer0);
      boolean boolean0 = jsonPointer0.equals(jsonPointer1);
      assertTrue(boolean0);
      assertEquals("", jsonPointer0.getMatchingProperty());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertFalse(jsonPointer1.matches());
      assertEquals((-1), jsonPointer1.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.EMPTY;
      boolean boolean0 = jsonPointer0.equals(jsonPointer0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.EMPTY;
      boolean boolean0 = jsonPointer0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.EMPTY;
      boolean boolean0 = jsonPointer0.equals("");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      // Undeclared exception!
      try { 
        JsonPointer._parseTail("r2i1s5Z8s1");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // For input string: \"2i1s5Z8s1\"
         //
         verifyException("java.lang.NumberFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("Invalid input: JSON Pointer expression must start with '/': \"");
      assertFalse(jsonPointer0.matches());
      assertEquals("nvalid input: JSON Pointer expression must start with '", jsonPointer0.getMatchingProperty());
      assertEquals("Invalid input: JSON Pointer expression must start with '/': \"", jsonPointer0.toString());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail(":~i(v");
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertEquals("~i(v", jsonPointer0.getMatchingProperty());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("N]a\"X]=C5C`G,j'+t#~");
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertEquals("]a\"X]=C5C`G,j'+t#~", jsonPointer0.getMatchingProperty());
      assertEquals("N]a\"X]=C5C`G,j'+t#~", jsonPointer0.toString());
      assertFalse(jsonPointer0.matches());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("W +d~1JUWIA9");
      assertEquals("W +d~1JUWIA9", jsonPointer0.toString());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertEquals(" +d/JUWIA9", jsonPointer0.getMatchingProperty());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseQuotedTail("Invalid input: JSON Pointer expression must start with '/': \"", 4);
      assertEquals("Invalid input: JSON Pointer expression must start with '/': \"", jsonPointer0.toString());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertEquals("nv~lid input: JSON Pointer expression must start with '", jsonPointer0.getMatchingProperty());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("hj%J!3=|Y~X~");
      assertEquals("hj%J!3=|Y~X~", jsonPointer0.toString());
      assertEquals("j%J!3=|Y~X~", jsonPointer0.getMatchingProperty());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("'KUZ,PQ?~-0&@~O-K");
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertEquals("KUZ,PQ?~-0&@~O-K", jsonPointer0.getMatchingProperty());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("Tr~0%uUH!ms*\"\"FqH");
      assertEquals("r~%uUH!ms*\"\"FqH", jsonPointer0.getMatchingProperty());
      assertFalse(jsonPointer0.matches());
  }
}
