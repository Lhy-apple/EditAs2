/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 02:23:49 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.CheckLevel;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.jscomp.PreprocessorSymbolTable;
import com.google.javascript.jscomp.ProcessClosurePrimitives;
import com.google.javascript.rhino.Node;
import java.util.Set;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ProcessClosurePrimitives_ESTest extends ProcessClosurePrimitives_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parseTestCode("goog.base");
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
      // Undeclared exception!
      try { 
        compiler0.parseTestCode("com.google.javascript.rhino.head.WrappedException");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: author=NOT_IMPLEMENTED and author=AUTHOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.OFF;
      ProcessClosurePrimitives processClosurePrimitives0 = new ProcessClosurePrimitives(compiler0, (PreprocessorSymbolTable) null, checkLevel0);
      Set<String> set0 = processClosurePrimitives0.getExportedVariableNames();
      assertEquals(0, set0.size());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parseTestCode("J.dbX34Ev=Zqzsq");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: author=NOT_IMPLEMENTED and author=AUTHOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parseTestCode("RJaL4,J=npqX+Ss");
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
      // Undeclared exception!
      try { 
        compiler0.parseTestCode("RJaL4,J=npqX+Ss");
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
      // Undeclared exception!
      try { 
        compiler0.parseTestCode("RJaL4,J=npqX+Ss");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: author=NOT_IMPLEMENTED and author=AUTHOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parseTestCode("com.google.javascript.rhino.head.WrappedException");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: author=NOT_IMPLEMENTED and author=AUTHOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }
}