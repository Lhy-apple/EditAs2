/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:25:16 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.CheckLevel;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.CompilerOptions;
import com.google.javascript.jscomp.DiagnosticGroup;
import com.google.javascript.jscomp.DiagnosticGroups;
import com.google.javascript.jscomp.DiagnosticType;
import com.google.javascript.jscomp.PhaseOptimizer;
import java.util.ArrayList;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DiagnosticGroups_ESTest extends DiagnosticGroups_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      DiagnosticType[] diagnosticTypeArray0 = new DiagnosticType[7];
      DiagnosticGroup diagnosticGroup0 = DiagnosticGroups.registerGroup("x+zSZ7Y", diagnosticTypeArray0);
      assertNotNull(diagnosticGroup0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      DiagnosticGroups diagnosticGroups0 = new DiagnosticGroups();
      // Undeclared exception!
      try { 
        diagnosticGroups0.getRegisteredGroups();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // null key
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      DiagnosticGroups diagnosticGroups0 = new DiagnosticGroups();
      List<String> list0 = PhaseOptimizer.OPTIMAL_ORDER;
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      // Undeclared exception!
      try { 
        diagnosticGroups0.setWarningLevels((CompilerOptions) null, list0, checkLevel0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // No warning class for name: removeUnreachableCode
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      DiagnosticGroups diagnosticGroups0 = new DiagnosticGroups();
      DiagnosticGroup diagnosticGroup0 = DiagnosticGroups.registerGroup("", diagnosticGroups0.STRICT_MODULE_DEP_CHECK);
      assertNotNull(diagnosticGroup0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      DiagnosticGroup[] diagnosticGroupArray0 = new DiagnosticGroup[0];
      DiagnosticGroup diagnosticGroup0 = DiagnosticGroups.registerGroup((String) null, diagnosticGroupArray0);
      assertNotNull(diagnosticGroup0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      DiagnosticGroups diagnosticGroups0 = compiler0.getDiagnosticGroups();
      ArrayList<String> arrayList0 = new ArrayList<String>();
      CheckLevel checkLevel0 = CheckLevel.OFF;
      diagnosticGroups0.setWarningLevels((CompilerOptions) null, arrayList0, checkLevel0);
      assertEquals(0, arrayList0.size());
  }
}