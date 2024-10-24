/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:18:24 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.NameAnalyzer;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class NameAnalyzer_ESTest extends NameAnalyzer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      NameAnalyzer nameAnalyzer0 = new NameAnalyzer(compiler0, true);
      // Undeclared exception!
      try { 
        compiler0.parseTestCode("RAW_SIZE");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: author=NOT_IMPLEMENTED and author=AUTHOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      NameAnalyzer nameAnalyzer0 = new NameAnalyzer(compiler0, true);
      String string0 = nameAnalyzer0.getHtmlReport();
      assertEquals("<html><body><style type=\"text/css\">body, td, p {font-family: Arial; font-size: 83%} ul {margin-top:2px; margin-left:0px; padding-left:1em;} li {margin-top:3px; margin-left:24px; padding-left:0px;padding-bottom: 4px}</style>OVERALL STATS<ul><li>Total Names: 0</li>\n<li>Total Classes: 0</li>\n<li>Total Static Functions: 0</li>\n<li>Referenced Names: 0</li>\n<li>Referenced Classes: 0</li>\n<li>Referenced Functions: 0</li>\n</ul>ALL NAMES<ul>\n</ul></body></html>", string0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("hS.~AaWw_");
      Node node1 = new Node(37, node0);
      NameAnalyzer nameAnalyzer0 = new NameAnalyzer(compiler0, false);
      nameAnalyzer0.process(node1, node1);
      assertFalse(node1.hasMoreThanOneChild());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("fY=Z");
      Node node1 = new Node(4, node0);
      NameAnalyzer nameAnalyzer0 = new NameAnalyzer(compiler0, true);
      nameAnalyzer0.process(node0, node1);
      assertTrue(node0.hasChildren());
      assertTrue(node0.hasOneChild());
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      NameAnalyzer nameAnalyzer0 = new NameAnalyzer(compiler0, true);
      // Undeclared exception!
      try { 
        compiler0.parseTestCode("com.google.javascript.jscomp.NameAnalyzer$JsNameRefNode");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: author=NOT_IMPLEMENTED and author=AUTHOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      NameAnalyzer nameAnalyzer0 = new NameAnalyzer(compiler0, false);
      // Undeclared exception!
      try { 
        compiler0.parseTestCode("window");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: author=NOT_IMPLEMENTED and author=AUTHOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      NameAnalyzer nameAnalyzer0 = new NameAnalyzer(compiler0, false);
      // Undeclared exception!
      try { 
        compiler0.parseTestCode("RAO_SIZE");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: author=NOT_IMPLEMENTED and author=AUTHOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }
}
