/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:08:34 GMT 2023
 */

package org.jsoup.select;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.select.Evaluator;
import org.jsoup.select.QueryParser;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class QueryParser_ESTest extends QueryParser_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      // Undeclared exception!
      try { 
        QueryParser.parse(":eq(");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Index must be numeric
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse("*,K");
      assertNotNull(evaluator0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      // Undeclared exception!
      try { 
        QueryParser.parse(":lt(");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Index must be numeric
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      // Undeclared exception!
      try { 
        QueryParser.parse("h#qn+{FXC3oJazW");
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Could not parse query '{FXC3oJazW': unexpected token at '{FXC3oJazW'
         //
         verifyException("org.jsoup.select.QueryParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      // Undeclared exception!
      try { 
        QueryParser.parse(":gt(");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Index must be numeric
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      // Undeclared exception!
      try { 
        QueryParser.parse(":has(");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // :has(el) subselect must not be empty
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse("org.jsoup.select.QueryParser");
      assertNotNull(evaluator0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      // Undeclared exception!
      try { 
        QueryParser.parse(":not(");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // :not(selector) subselect must not be empty
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      // Undeclared exception!
      try { 
        QueryParser.parse(",yvF");
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Unknown combinator: ,
         //
         verifyException("org.jsoup.select.QueryParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      // Undeclared exception!
      try { 
        QueryParser.parse(":matches(regex) query must not be em`ty");
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Could not parse query 'em`ty': unexpected token at '`ty'
         //
         verifyException("org.jsoup.select.QueryParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      // Undeclared exception!
      try { 
        QueryParser.parse("E5pC,>p -VSeg68Xb");
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Could not parse query '-VSeg68Xb': unexpected token at '-VSeg68Xb'
         //
         verifyException("org.jsoup.select.QueryParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse("w+MRA[nR[?`Z\"8]u");
      assertNotNull(evaluator0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse("8IS~[k'Y4a7mm'0");
      assertNotNull(evaluator0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      // Undeclared exception!
      try { 
        QueryParser.parse("u_>i6M4{4F(k");
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Could not parse query 'i6M4{4F(k)': unexpected token at '{4F(k)'
         //
         verifyException("org.jsoup.select.QueryParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse(":contains(%s");
      assertNotNull(evaluator0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse(":containsOwn(%s");
      assertNotNull(evaluator0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse(":matchesOwn(%s");
      assertNotNull(evaluator0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse("4|QX");
      assertNotNull(evaluator0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse("[%s$=%s]");
      assertNotNull(evaluator0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse("[^%s]");
      assertNotNull(evaluator0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse("x-2[?'#%UbTY=B&Op");
      assertNotNull(evaluator0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse("[%s!=%s]");
      assertNotNull(evaluator0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse("[%s^=%s]");
      assertNotNull(evaluator0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse("[%s~=%s]");
      assertNotNull(evaluator0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse("[%s*=%s]");
      assertNotNull(evaluator0);
  }
}