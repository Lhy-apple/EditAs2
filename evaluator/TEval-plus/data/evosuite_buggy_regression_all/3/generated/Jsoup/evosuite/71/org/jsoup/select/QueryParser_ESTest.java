/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:53:15 GMT 2023
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
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Index must be numeric
         //
         verifyException("org.jsoup.select.QueryParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse("Qr*5Bcmo");
      assertNotNull(evaluator0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      // Undeclared exception!
      try { 
        QueryParser.parse(":containsData(");
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Did not find balanced marker at ''
         //
         verifyException("org.jsoup.select.QueryParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      // Undeclared exception!
      try { 
        QueryParser.parse(":lt(");
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Index must be numeric
         //
         verifyException("org.jsoup.select.QueryParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      // Undeclared exception!
      try { 
        QueryParser.parse("xCzjYboV#");
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // String must not be empty
         //
         verifyException("org.jsoup.select.QueryParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      // Undeclared exception!
      try { 
        QueryParser.parse(":gt(");
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Index must be numeric
         //
         verifyException("org.jsoup.select.QueryParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      // Undeclared exception!
      try { 
        QueryParser.parse(":has(");
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Did not find balanced marker at ''
         //
         verifyException("org.jsoup.select.QueryParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse("6u.iQ~U");
      assertNotNull(evaluator0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      // Undeclared exception!
      try { 
        QueryParser.parse(":not(");
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Did not find balanced marker at ''
         //
         verifyException("org.jsoup.select.QueryParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse("y,QxH_RKAhv>E");
      assertNotNull(evaluator0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse("KR,7Lwj,xvf");
      assertNotNull(evaluator0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse("q8h+d8uUTm,B");
      assertNotNull(evaluator0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      // Undeclared exception!
      try { 
        QueryParser.parse("+lk(2!\"\"`\"h-/0w%bqq");
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Did not find balanced marker at '2!\"\"`'
         //
         verifyException("org.jsoup.select.QueryParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      // Undeclared exception!
      try { 
        QueryParser.parse("kh,s'VHqNW[2m='t6");
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Did not find balanced marker at '2m='
         //
         verifyException("org.jsoup.select.QueryParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse("*|xml");
      assertNotNull(evaluator0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse("[%s*=%s]");
      assertNotNull(evaluator0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse(":contains(text) query must not be empty");
      assertNotNull(evaluator0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse(":containsOwn(%s)");
      assertNotNull(evaluator0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse(":matches(regex) query must not be empty");
      assertNotNull(evaluator0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse(":matchesOwn(%s)");
      assertNotNull(evaluator0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      // Undeclared exception!
      try { 
        QueryParser.parse(":nth-child(");
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Could not parse nth-index '': unexpected format
         //
         verifyException("org.jsoup.select.QueryParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      // Undeclared exception!
      try { 
        QueryParser.parse(":nth-last-child(");
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Could not parse nth-index '': unexpected format
         //
         verifyException("org.jsoup.select.QueryParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      // Undeclared exception!
      try { 
        QueryParser.parse(":nth-of-type(");
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Could not parse nth-index '': unexpected format
         //
         verifyException("org.jsoup.select.QueryParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      // Undeclared exception!
      try { 
        QueryParser.parse(":nth-last-of-type(");
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Could not parse nth-index '': unexpected format
         //
         verifyException("org.jsoup.select.QueryParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse(":first-child");
      assertEquals(":first-child", evaluator0.toString());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse(":last-child");
      assertEquals(":last-child", evaluator0.toString());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse(":first-of-type");
      assertEquals(":first-of-type", evaluator0.toString());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse(":last-of-type");
      assertEquals(":last-of-type", evaluator0.toString());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse(":only-child");
      assertEquals(":only-child", evaluator0.toString());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse(":only-of-type");
      assertEquals(":only-of-type", evaluator0.toString());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse(":empty");
      assertEquals(":empty", evaluator0.toString());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse(":root");
      assertEquals(":root", evaluator0.toString());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      // Undeclared exception!
      try { 
        QueryParser.parse("MDd, p|ZM!W,k+7n");
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Could not parse query 'p|ZM!W': unexpected token at '!W'
         //
         verifyException("org.jsoup.select.QueryParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse("[%s]");
      assertNotNull(evaluator0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse("[^%s]");
      assertNotNull(evaluator0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse("[%s=s]");
      assertNotNull(evaluator0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse("[%s!=%s]");
      assertNotNull(evaluator0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse("[%s^=%s]");
      assertNotNull(evaluator0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse("[%s$=%s]");
      assertNotNull(evaluator0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Evaluator evaluator0 = QueryParser.parse("[%s~=%s]");
      assertNotNull(evaluator0);
  }
}