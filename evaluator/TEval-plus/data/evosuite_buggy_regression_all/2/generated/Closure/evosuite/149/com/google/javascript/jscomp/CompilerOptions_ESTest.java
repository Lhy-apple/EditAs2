/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:40:27 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.ClosureCodingConvention;
import com.google.javascript.jscomp.CompilerOptions;
import com.google.javascript.jscomp.DiagnosticGroup;
import com.google.javascript.jscomp.DiagnosticType;
import com.google.javascript.jscomp.ShowByPathWarningsGuard;
import com.google.javascript.jscomp.TypeCheck;
import com.google.javascript.rhino.Node;
import java.util.Map;
import java.util.Set;
import java.util.Vector;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CompilerOptions_ESTest extends CompilerOptions_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compilerOptions0.setRemoveAbstractMethods(false);
      assertFalse(compilerOptions0.strictMessageReplacement);
      assertFalse(compilerOptions0.inferTypesInGlobalScope);
      assertFalse(compilerOptions0.checkControlStructures);
      assertFalse(compilerOptions0.checkUnusedPropertiesEarly);
      assertFalse(compilerOptions0.allowLegacyJsMessages);
      assertFalse(compilerOptions0.checkTypes);
      assertFalse(compilerOptions0.checkSuspiciousCode);
      assertFalse(compilerOptions0.tightenTypes);
      assertFalse(compilerOptions0.checkSymbols);
      assertFalse(compilerOptions0.checkDuplicateMessages);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compilerOptions0.getWarningsGuard();
      assertFalse(compilerOptions0.checkSuspiciousCode);
      assertFalse(compilerOptions0.tightenTypes);
      assertFalse(compilerOptions0.inferTypesInGlobalScope);
      assertFalse(compilerOptions0.allowLegacyJsMessages);
      assertFalse(compilerOptions0.strictMessageReplacement);
      assertFalse(compilerOptions0.checkUnusedPropertiesEarly);
      assertFalse(compilerOptions0.checkDuplicateMessages);
      assertFalse(compilerOptions0.checkControlStructures);
      assertFalse(compilerOptions0.checkSymbols);
      assertFalse(compilerOptions0.checkTypes);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compilerOptions0.setNameAnonymousFunctionsOnly(false);
      assertFalse(compilerOptions0.checkUnusedPropertiesEarly);
      assertFalse(compilerOptions0.strictMessageReplacement);
      assertFalse(compilerOptions0.inferTypesInGlobalScope);
      assertFalse(compilerOptions0.checkDuplicateMessages);
      assertFalse(compilerOptions0.checkTypes);
      assertFalse(compilerOptions0.checkSymbols);
      assertFalse(compilerOptions0.tightenTypes);
      assertFalse(compilerOptions0.checkSuspiciousCode);
      assertFalse(compilerOptions0.allowLegacyJsMessages);
      assertFalse(compilerOptions0.checkControlStructures);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      Vector<String> vector0 = new Vector<String>(4);
      compilerOptions0.setReplaceStringsConfiguration((String) null, vector0);
      assertFalse(compilerOptions0.checkControlStructures);
      assertFalse(compilerOptions0.tightenTypes);
      assertFalse(compilerOptions0.checkSuspiciousCode);
      assertFalse(compilerOptions0.checkTypes);
      assertFalse(compilerOptions0.checkDuplicateMessages);
      assertFalse(compilerOptions0.checkSymbols);
      assertFalse(compilerOptions0.strictMessageReplacement);
      assertFalse(compilerOptions0.inferTypesInGlobalScope);
      assertFalse(compilerOptions0.allowLegacyJsMessages);
      assertFalse(compilerOptions0.checkUnusedPropertiesEarly);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compilerOptions0.setSummaryDetailLevel(0);
      assertFalse(compilerOptions0.checkSymbols);
      assertFalse(compilerOptions0.inferTypesInGlobalScope);
      assertFalse(compilerOptions0.strictMessageReplacement);
      assertFalse(compilerOptions0.allowLegacyJsMessages);
      assertFalse(compilerOptions0.checkSuspiciousCode);
      assertFalse(compilerOptions0.tightenTypes);
      assertFalse(compilerOptions0.checkControlStructures);
      assertFalse(compilerOptions0.checkUnusedPropertiesEarly);
      assertFalse(compilerOptions0.checkTypes);
      assertFalse(compilerOptions0.checkDuplicateMessages);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compilerOptions0.setDefineToNumberLiteral((String) null, 19);
      Map<String, Node> map0 = compilerOptions0.getDefineReplacements();
      assertFalse(compilerOptions0.strictMessageReplacement);
      assertFalse(compilerOptions0.checkSuspiciousCode);
      assertFalse(compilerOptions0.tightenTypes);
      assertFalse(compilerOptions0.allowLegacyJsMessages);
      assertFalse(compilerOptions0.checkDuplicateMessages);
      assertFalse(compilerOptions0.inferTypesInGlobalScope);
      assertFalse(compilerOptions0.checkControlStructures);
      assertFalse(compilerOptions0.checkUnusedPropertiesEarly);
      assertFalse(compilerOptions0.checkSymbols);
      assertFalse(compilerOptions0.checkTypes);
      assertFalse(map0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compilerOptions0.setColorizeErrorOutput(false);
      assertFalse(compilerOptions0.strictMessageReplacement);
      assertFalse(compilerOptions0.checkSymbols);
      assertFalse(compilerOptions0.shouldColorizeErrorOutput());
      assertFalse(compilerOptions0.allowLegacyJsMessages);
      assertFalse(compilerOptions0.checkTypes);
      assertFalse(compilerOptions0.checkUnusedPropertiesEarly);
      assertFalse(compilerOptions0.inferTypesInGlobalScope);
      assertFalse(compilerOptions0.checkControlStructures);
      assertFalse(compilerOptions0.checkDuplicateMessages);
      assertFalse(compilerOptions0.tightenTypes);
      assertFalse(compilerOptions0.checkSuspiciousCode);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compilerOptions0.setLooseTypes(false);
      assertFalse(compilerOptions0.checkDuplicateMessages);
      assertFalse(compilerOptions0.strictMessageReplacement);
      assertFalse(compilerOptions0.checkSuspiciousCode);
      assertFalse(compilerOptions0.tightenTypes);
      assertFalse(compilerOptions0.checkTypes);
      assertFalse(compilerOptions0.checkSymbols);
      assertFalse(compilerOptions0.inferTypesInGlobalScope);
      assertFalse(compilerOptions0.checkUnusedPropertiesEarly);
      assertFalse(compilerOptions0.allowLegacyJsMessages);
      assertFalse(compilerOptions0.checkControlStructures);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compilerOptions0.setProcessObjectPropertyString(false);
      assertFalse(compilerOptions0.inferTypesInGlobalScope);
      assertFalse(compilerOptions0.checkSymbols);
      assertFalse(compilerOptions0.strictMessageReplacement);
      assertFalse(compilerOptions0.checkSuspiciousCode);
      assertFalse(compilerOptions0.tightenTypes);
      assertFalse(compilerOptions0.allowLegacyJsMessages);
      assertFalse(compilerOptions0.checkControlStructures);
      assertFalse(compilerOptions0.checkUnusedPropertiesEarly);
      assertFalse(compilerOptions0.checkTypes);
      assertFalse(compilerOptions0.checkDuplicateMessages);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      boolean boolean0 = compilerOptions0.isExternExportsEnabled();
      assertFalse(compilerOptions0.checkSymbols);
      assertFalse(compilerOptions0.checkTypes);
      assertFalse(compilerOptions0.allowLegacyJsMessages);
      assertFalse(compilerOptions0.checkControlStructures);
      assertFalse(boolean0);
      assertFalse(compilerOptions0.inferTypesInGlobalScope);
      assertFalse(compilerOptions0.checkUnusedPropertiesEarly);
      assertFalse(compilerOptions0.checkDuplicateMessages);
      assertFalse(compilerOptions0.strictMessageReplacement);
      assertFalse(compilerOptions0.checkSuspiciousCode);
      assertFalse(compilerOptions0.tightenTypes);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compilerOptions0.setChainCalls(false);
      assertFalse(compilerOptions0.strictMessageReplacement);
      assertFalse(compilerOptions0.checkSymbols);
      assertFalse(compilerOptions0.inferTypesInGlobalScope);
      assertFalse(compilerOptions0.checkTypes);
      assertFalse(compilerOptions0.checkDuplicateMessages);
      assertFalse(compilerOptions0.checkControlStructures);
      assertFalse(compilerOptions0.checkSuspiciousCode);
      assertFalse(compilerOptions0.tightenTypes);
      assertFalse(compilerOptions0.checkUnusedPropertiesEarly);
      assertFalse(compilerOptions0.allowLegacyJsMessages);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compilerOptions0.enableExternExports(false);
      assertFalse(compilerOptions0.isExternExportsEnabled());
      assertFalse(compilerOptions0.checkDuplicateMessages);
      assertFalse(compilerOptions0.strictMessageReplacement);
      assertFalse(compilerOptions0.tightenTypes);
      assertFalse(compilerOptions0.checkSuspiciousCode);
      assertFalse(compilerOptions0.checkTypes);
      assertFalse(compilerOptions0.checkSymbols);
      assertFalse(compilerOptions0.inferTypesInGlobalScope);
      assertFalse(compilerOptions0.checkControlStructures);
      assertFalse(compilerOptions0.allowLegacyJsMessages);
      assertFalse(compilerOptions0.checkUnusedPropertiesEarly);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compilerOptions0.setRewriteNewDateGoogNow(true);
      assertFalse(compilerOptions0.checkSymbols);
      assertFalse(compilerOptions0.strictMessageReplacement);
      assertFalse(compilerOptions0.inferTypesInGlobalScope);
      assertFalse(compilerOptions0.allowLegacyJsMessages);
      assertFalse(compilerOptions0.checkSuspiciousCode);
      assertFalse(compilerOptions0.tightenTypes);
      assertFalse(compilerOptions0.checkControlStructures);
      assertFalse(compilerOptions0.checkUnusedPropertiesEarly);
      assertFalse(compilerOptions0.checkTypes);
      assertFalse(compilerOptions0.checkDuplicateMessages);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      compilerOptions0.setCodingConvention(closureCodingConvention0);
      assertFalse(compilerOptions0.checkControlStructures);
      assertFalse(compilerOptions0.checkDuplicateMessages);
      assertFalse(compilerOptions0.tightenTypes);
      assertFalse(compilerOptions0.checkSuspiciousCode);
      assertFalse(compilerOptions0.strictMessageReplacement);
      assertFalse(compilerOptions0.checkSymbols);
      assertFalse(compilerOptions0.allowLegacyJsMessages);
      assertFalse(compilerOptions0.checkTypes);
      assertFalse(compilerOptions0.checkUnusedPropertiesEarly);
      assertFalse(compilerOptions0.inferTypesInGlobalScope);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compilerOptions0.setDefineToStringLiteral("t\"Xt_ise8", "t\"Xt_ise8");
      assertFalse(compilerOptions0.checkSymbols);
      assertFalse(compilerOptions0.strictMessageReplacement);
      assertFalse(compilerOptions0.inferTypesInGlobalScope);
      assertFalse(compilerOptions0.checkTypes);
      assertFalse(compilerOptions0.checkDuplicateMessages);
      assertFalse(compilerOptions0.checkControlStructures);
      assertFalse(compilerOptions0.checkUnusedPropertiesEarly);
      assertFalse(compilerOptions0.tightenTypes);
      assertFalse(compilerOptions0.checkSuspiciousCode);
      assertFalse(compilerOptions0.allowLegacyJsMessages);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      // Undeclared exception!
      try { 
        compilerOptions0.setIdGenerators((Set<String>) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.collect.Sets", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compilerOptions0.enableRuntimeTypeCheck((String) null);
      assertFalse(compilerOptions0.inferTypesInGlobalScope);
      assertFalse(compilerOptions0.allowLegacyJsMessages);
      assertFalse(compilerOptions0.checkSymbols);
      assertFalse(compilerOptions0.checkTypes);
      assertFalse(compilerOptions0.checkUnusedPropertiesEarly);
      assertFalse(compilerOptions0.checkDuplicateMessages);
      assertFalse(compilerOptions0.checkControlStructures);
      assertFalse(compilerOptions0.strictMessageReplacement);
      assertFalse(compilerOptions0.checkSuspiciousCode);
      assertFalse(compilerOptions0.tightenTypes);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compilerOptions0.setDefineToDoubleLiteral("// Input %num%", 19);
      Map<String, Node> map0 = compilerOptions0.getDefineReplacements();
      assertFalse(compilerOptions0.checkUnusedPropertiesEarly);
      assertFalse(compilerOptions0.checkDuplicateMessages);
      assertFalse(compilerOptions0.checkControlStructures);
      assertFalse(map0.isEmpty());
      assertFalse(compilerOptions0.checkSymbols);
      assertFalse(compilerOptions0.checkTypes);
      assertFalse(compilerOptions0.tightenTypes);
      assertFalse(compilerOptions0.checkSuspiciousCode);
      assertFalse(compilerOptions0.inferTypesInGlobalScope);
      assertFalse(compilerOptions0.allowLegacyJsMessages);
      assertFalse(compilerOptions0.strictMessageReplacement);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      CompilerOptions compilerOptions1 = (CompilerOptions)compilerOptions0.clone();
      assertFalse(compilerOptions1.checkControlStructures);
      assertFalse(compilerOptions1.inferTypesInGlobalScope);
      assertFalse(compilerOptions1.checkUnusedPropertiesEarly);
      assertFalse(compilerOptions1.tightenTypes);
      assertFalse(compilerOptions1.allowLegacyJsMessages);
      assertFalse(compilerOptions1.checkSuspiciousCode);
      assertFalse(compilerOptions1.checkTypes);
      assertFalse(compilerOptions1.checkSymbols);
      assertNotSame(compilerOptions1, compilerOptions0);
      assertFalse(compilerOptions1.strictMessageReplacement);
      assertFalse(compilerOptions1.checkDuplicateMessages);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compilerOptions0.setDefineToBooleanLiteral("", false);
      compilerOptions0.getDefineReplacements();
      assertFalse(compilerOptions0.allowLegacyJsMessages);
      assertFalse(compilerOptions0.checkTypes);
      assertFalse(compilerOptions0.checkSuspiciousCode);
      assertFalse(compilerOptions0.tightenTypes);
      assertFalse(compilerOptions0.checkControlStructures);
      assertFalse(compilerOptions0.strictMessageReplacement);
      assertFalse(compilerOptions0.checkSymbols);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compilerOptions0.skipAllCompilerPasses();
      assertFalse(compilerOptions0.checkSuspiciousCode);
      assertFalse(compilerOptions0.allowLegacyJsMessages);
      assertFalse(compilerOptions0.checkTypes);
      assertFalse(compilerOptions0.tightenTypes);
      assertFalse(compilerOptions0.checkSymbols);
      assertFalse(compilerOptions0.checkDuplicateMessages);
      assertFalse(compilerOptions0.strictMessageReplacement);
      assertFalse(compilerOptions0.checkControlStructures);
      assertFalse(compilerOptions0.inferTypesInGlobalScope);
      assertFalse(compilerOptions0.checkUnusedPropertiesEarly);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compilerOptions0.disableRuntimeTypeCheck();
      assertFalse(compilerOptions0.tightenTypes);
      assertFalse(compilerOptions0.checkSuspiciousCode);
      assertFalse(compilerOptions0.checkControlStructures);
      assertFalse(compilerOptions0.allowLegacyJsMessages);
      assertFalse(compilerOptions0.inferTypesInGlobalScope);
      assertFalse(compilerOptions0.checkDuplicateMessages);
      assertFalse(compilerOptions0.checkSymbols);
      assertFalse(compilerOptions0.checkTypes);
      assertFalse(compilerOptions0.strictMessageReplacement);
      assertFalse(compilerOptions0.checkUnusedPropertiesEarly);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compilerOptions0.setCollapsePropertiesOnExternTypes(true);
      assertFalse(compilerOptions0.checkSymbols);
      assertFalse(compilerOptions0.inferTypesInGlobalScope);
      assertFalse(compilerOptions0.checkControlStructures);
      assertFalse(compilerOptions0.checkUnusedPropertiesEarly);
      assertFalse(compilerOptions0.allowLegacyJsMessages);
      assertFalse(compilerOptions0.tightenTypes);
      assertFalse(compilerOptions0.checkSuspiciousCode);
      assertFalse(compilerOptions0.checkTypes);
      assertFalse(compilerOptions0.checkDuplicateMessages);
      assertFalse(compilerOptions0.strictMessageReplacement);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compilerOptions0.setManageClosureDependencies(false);
      assertFalse(compilerOptions0.checkUnusedPropertiesEarly);
      assertFalse(compilerOptions0.strictMessageReplacement);
      assertFalse(compilerOptions0.checkSymbols);
      assertFalse(compilerOptions0.checkDuplicateMessages);
      assertFalse(compilerOptions0.tightenTypes);
      assertFalse(compilerOptions0.checkSuspiciousCode);
      assertFalse(compilerOptions0.checkTypes);
      assertFalse(compilerOptions0.allowLegacyJsMessages);
      assertFalse(compilerOptions0.inferTypesInGlobalScope);
      assertFalse(compilerOptions0.checkControlStructures);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      boolean boolean0 = compilerOptions0.shouldColorizeErrorOutput();
      assertFalse(compilerOptions0.checkUnusedPropertiesEarly);
      assertFalse(compilerOptions0.strictMessageReplacement);
      assertFalse(compilerOptions0.checkSymbols);
      assertFalse(compilerOptions0.checkDuplicateMessages);
      assertFalse(compilerOptions0.checkTypes);
      assertFalse(compilerOptions0.allowLegacyJsMessages);
      assertFalse(boolean0);
      assertFalse(compilerOptions0.inferTypesInGlobalScope);
      assertFalse(compilerOptions0.checkSuspiciousCode);
      assertFalse(compilerOptions0.tightenTypes);
      assertFalse(compilerOptions0.checkControlStructures);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compilerOptions0.getCodingConvention();
      assertFalse(compilerOptions0.inferTypesInGlobalScope);
      assertFalse(compilerOptions0.checkSymbols);
      assertFalse(compilerOptions0.checkSuspiciousCode);
      assertFalse(compilerOptions0.checkTypes);
      assertFalse(compilerOptions0.tightenTypes);
      assertFalse(compilerOptions0.strictMessageReplacement);
      assertFalse(compilerOptions0.checkDuplicateMessages);
      assertFalse(compilerOptions0.checkUnusedPropertiesEarly);
      assertFalse(compilerOptions0.allowLegacyJsMessages);
      assertFalse(compilerOptions0.checkControlStructures);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compilerOptions0.setRenamingPolicy(compilerOptions0.variableRenaming, compilerOptions0.propertyRenaming);
      assertFalse(compilerOptions0.checkControlStructures);
      assertFalse(compilerOptions0.checkDuplicateMessages);
      assertFalse(compilerOptions0.tightenTypes);
      assertFalse(compilerOptions0.checkSuspiciousCode);
      assertFalse(compilerOptions0.strictMessageReplacement);
      assertFalse(compilerOptions0.checkSymbols);
      assertFalse(compilerOptions0.checkTypes);
      assertFalse(compilerOptions0.allowLegacyJsMessages);
      assertFalse(compilerOptions0.checkUnusedPropertiesEarly);
      assertFalse(compilerOptions0.inferTypesInGlobalScope);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      DiagnosticGroup diagnosticGroup0 = TypeCheck.ALL_DIAGNOSTICS;
      compilerOptions0.setWarningLevel(diagnosticGroup0, compilerOptions0.checkFunctions);
      assertFalse(compilerOptions0.checkTypes);
      assertFalse(compilerOptions0.allowLegacyJsMessages);
      assertFalse(compilerOptions0.tightenTypes);
      assertFalse(compilerOptions0.checkSuspiciousCode);
      assertFalse(compilerOptions0.checkSymbols);
      assertFalse(compilerOptions0.checkDuplicateMessages);
      assertFalse(compilerOptions0.strictMessageReplacement);
      assertFalse(compilerOptions0.checkUnusedPropertiesEarly);
      assertFalse(compilerOptions0.inferTypesInGlobalScope);
      assertFalse(compilerOptions0.checkControlStructures);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compilerOptions0.setDefineToBooleanLiteral("", true);
      compilerOptions0.getDefineReplacements();
      assertFalse(compilerOptions0.checkDuplicateMessages);
      assertFalse(compilerOptions0.inferTypesInGlobalScope);
      assertFalse(compilerOptions0.checkSuspiciousCode);
      assertFalse(compilerOptions0.tightenTypes);
      assertFalse(compilerOptions0.checkTypes);
      assertFalse(compilerOptions0.allowLegacyJsMessages);
      assertFalse(compilerOptions0.strictMessageReplacement);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      DiagnosticType diagnosticType0 = TypeCheck.WRONG_ARGUMENT_COUNT;
      DiagnosticGroup diagnosticGroup0 = DiagnosticGroup.forType(diagnosticType0);
      boolean boolean0 = compilerOptions0.enables(diagnosticGroup0);
      assertFalse(compilerOptions0.checkControlStructures);
      assertFalse(compilerOptions0.checkDuplicateMessages);
      assertFalse(compilerOptions0.checkTypes);
      assertFalse(compilerOptions0.allowLegacyJsMessages);
      assertFalse(compilerOptions0.checkUnusedPropertiesEarly);
      assertFalse(boolean0);
      assertFalse(compilerOptions0.inferTypesInGlobalScope);
      assertFalse(compilerOptions0.checkSymbols);
      assertFalse(compilerOptions0.checkSuspiciousCode);
      assertFalse(compilerOptions0.tightenTypes);
      assertFalse(compilerOptions0.strictMessageReplacement);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      ShowByPathWarningsGuard.ShowType showByPathWarningsGuard_ShowType0 = ShowByPathWarningsGuard.ShowType.INCLUDE;
      String[] stringArray0 = new String[4];
      ShowByPathWarningsGuard showByPathWarningsGuard0 = new ShowByPathWarningsGuard(stringArray0, showByPathWarningsGuard_ShowType0);
      compilerOptions0.addWarningsGuard(showByPathWarningsGuard0);
      boolean boolean0 = compilerOptions0.enables((DiagnosticGroup) null);
      assertFalse(boolean0);
      assertFalse(compilerOptions0.checkDuplicateMessages);
      assertFalse(compilerOptions0.strictMessageReplacement);
      assertFalse(compilerOptions0.checkUnusedPropertiesEarly);
      assertFalse(compilerOptions0.checkControlStructures);
      assertFalse(compilerOptions0.checkSymbols);
      assertFalse(compilerOptions0.checkTypes);
      assertFalse(compilerOptions0.checkSuspiciousCode);
      assertFalse(compilerOptions0.tightenTypes);
      assertFalse(compilerOptions0.inferTypesInGlobalScope);
      assertFalse(compilerOptions0.allowLegacyJsMessages);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      boolean boolean0 = compilerOptions0.disables((DiagnosticGroup) null);
      assertFalse(compilerOptions0.checkSuspiciousCode);
      assertFalse(compilerOptions0.checkTypes);
      assertFalse(compilerOptions0.tightenTypes);
      assertFalse(compilerOptions0.allowLegacyJsMessages);
      assertFalse(compilerOptions0.strictMessageReplacement);
      assertFalse(compilerOptions0.checkDuplicateMessages);
      assertFalse(compilerOptions0.checkControlStructures);
      assertFalse(compilerOptions0.checkSymbols);
      assertFalse(boolean0);
      assertFalse(compilerOptions0.checkUnusedPropertiesEarly);
      assertFalse(compilerOptions0.inferTypesInGlobalScope);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      DiagnosticType diagnosticType0 = TypeCheck.WRONG_ARGUMENT_COUNT;
      ShowByPathWarningsGuard.ShowType showByPathWarningsGuard_ShowType0 = ShowByPathWarningsGuard.ShowType.INCLUDE;
      String[] stringArray0 = new String[4];
      ShowByPathWarningsGuard showByPathWarningsGuard0 = new ShowByPathWarningsGuard(stringArray0, showByPathWarningsGuard_ShowType0);
      DiagnosticGroup diagnosticGroup0 = DiagnosticGroup.forType(diagnosticType0);
      compilerOptions0.addWarningsGuard(showByPathWarningsGuard0);
      boolean boolean0 = compilerOptions0.disables(diagnosticGroup0);
      assertFalse(compilerOptions0.checkTypes);
      assertFalse(compilerOptions0.checkSymbols);
      assertFalse(compilerOptions0.allowLegacyJsMessages);
      assertFalse(compilerOptions0.checkUnusedPropertiesEarly);
      assertFalse(compilerOptions0.inferTypesInGlobalScope);
      assertFalse(compilerOptions0.checkControlStructures);
      assertFalse(compilerOptions0.checkDuplicateMessages);
      assertFalse(boolean0);
      assertFalse(compilerOptions0.checkSuspiciousCode);
      assertFalse(compilerOptions0.tightenTypes);
      assertFalse(compilerOptions0.strictMessageReplacement);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      ShowByPathWarningsGuard.ShowType showByPathWarningsGuard_ShowType0 = ShowByPathWarningsGuard.ShowType.INCLUDE;
      String[] stringArray0 = new String[4];
      ShowByPathWarningsGuard showByPathWarningsGuard0 = new ShowByPathWarningsGuard(stringArray0, showByPathWarningsGuard_ShowType0);
      compilerOptions0.addWarningsGuard(showByPathWarningsGuard0);
      DiagnosticType[] diagnosticTypeArray0 = new DiagnosticType[0];
      DiagnosticGroup diagnosticGroup0 = new DiagnosticGroup(diagnosticTypeArray0);
      boolean boolean0 = compilerOptions0.disables(diagnosticGroup0);
      assertFalse(compilerOptions0.strictMessageReplacement);
      assertFalse(compilerOptions0.checkSymbols);
      assertFalse(compilerOptions0.tightenTypes);
      assertFalse(compilerOptions0.checkSuspiciousCode);
      assertFalse(compilerOptions0.checkDuplicateMessages);
      assertFalse(compilerOptions0.checkTypes);
      assertTrue(boolean0);
      assertFalse(compilerOptions0.inferTypesInGlobalScope);
      assertFalse(compilerOptions0.checkControlStructures);
      assertFalse(compilerOptions0.allowLegacyJsMessages);
      assertFalse(compilerOptions0.checkUnusedPropertiesEarly);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      ShowByPathWarningsGuard showByPathWarningsGuard0 = new ShowByPathWarningsGuard("// Input %num%");
      compilerOptions0.addWarningsGuard(showByPathWarningsGuard0);
      compilerOptions0.addWarningsGuard(showByPathWarningsGuard0);
      assertFalse(compilerOptions0.checkControlStructures);
      assertFalse(compilerOptions0.allowLegacyJsMessages);
      assertFalse(compilerOptions0.checkDuplicateMessages);
      assertFalse(compilerOptions0.checkUnusedPropertiesEarly);
      assertFalse(compilerOptions0.inferTypesInGlobalScope);
      assertFalse(compilerOptions0.checkSymbols);
      assertFalse(compilerOptions0.checkTypes);
      assertFalse(compilerOptions0.strictMessageReplacement);
      assertFalse(compilerOptions0.checkSuspiciousCode);
      assertFalse(compilerOptions0.tightenTypes);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      boolean boolean0 = compilerOptions0.tracer.isOn();
      assertFalse(compilerOptions0.inferTypesInGlobalScope);
      assertFalse(compilerOptions0.checkTypes);
      assertFalse(compilerOptions0.checkSymbols);
      assertFalse(compilerOptions0.checkUnusedPropertiesEarly);
      assertFalse(boolean0);
      assertFalse(compilerOptions0.allowLegacyJsMessages);
      assertFalse(compilerOptions0.checkControlStructures);
      assertFalse(compilerOptions0.checkDuplicateMessages);
      assertFalse(compilerOptions0.tightenTypes);
      assertFalse(compilerOptions0.checkSuspiciousCode);
      assertFalse(compilerOptions0.strictMessageReplacement);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      CompilerOptions.TracerMode compilerOptions_TracerMode0 = CompilerOptions.TracerMode.ALL;
      boolean boolean0 = compilerOptions_TracerMode0.isOn();
      assertTrue(boolean0);
  }
}