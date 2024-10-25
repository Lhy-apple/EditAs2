/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:09:04 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import com.google.javascript.jscomp.CheckPropertyOrder;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.DiagnosticType;
import com.google.javascript.jscomp.JSError;
import com.google.javascript.jscomp.LightweightMessageFormatter;
import com.google.javascript.jscomp.ProcessClosurePrimitives;
import com.google.javascript.jscomp.Region;
import com.google.javascript.jscomp.SimpleRegion;
import com.google.javascript.jscomp.TypedScopeCreator;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class LightweightMessageFormatter_ESTest extends LightweightMessageFormatter_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      LightweightMessageFormatter.LineNumberingFormatter lightweightMessageFormatter_LineNumberingFormatter0 = new LightweightMessageFormatter.LineNumberingFormatter();
      SimpleRegion simpleRegion0 = new SimpleRegion(22, 22, "^\n");
      String string0 = lightweightMessageFormatter_LineNumberingFormatter0.formatRegion(simpleRegion0);
      assertNotNull(string0);
      assertEquals("  22| ^", string0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      DiagnosticType diagnosticType0 = ProcessClosurePrimitives.XMODULE_REQUIRE_ERROR;
      JSError jSError0 = JSError.make(diagnosticType0, (String[]) null);
      Compiler compiler0 = new Compiler();
      LightweightMessageFormatter lightweightMessageFormatter0 = new LightweightMessageFormatter(compiler0);
      String string0 = lightweightMessageFormatter0.formatWarning(jSError0);
      assertEquals("WARNING - namespace \"{0}\" provided in module {1} but required in module {2}\n", string0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      LightweightMessageFormatter lightweightMessageFormatter0 = LightweightMessageFormatter.withoutSource();
      DiagnosticType diagnosticType0 = CheckPropertyOrder.UNEQUAL_PROPERTIES;
      JSError jSError0 = JSError.make(diagnosticType0, (String[]) null);
      String string0 = lightweightMessageFormatter0.formatError(jSError0);
      assertEquals("ERROR - different control paths produce different (ordered) property lists: {0} vs. {1}\n", string0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      LightweightMessageFormatter lightweightMessageFormatter0 = LightweightMessageFormatter.withoutSource();
      DiagnosticType diagnosticType0 = TypedScopeCreator.CONSTRUCTOR_EXPECTED;
      JSError jSError0 = JSError.make("JSC_REFLECT_CONSTRUCTOR_EXPECTED", (Node) null, diagnosticType0, (String[]) null);
      String string0 = lightweightMessageFormatter0.formatWarning(jSError0);
      assertEquals("JSC_REFLECT_CONSTRUCTOR_EXPECTED: WARNING - Constructor expected as first argument\n", string0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      LightweightMessageFormatter lightweightMessageFormatter0 = LightweightMessageFormatter.withoutSource();
      Node node0 = Node.newNumber((double) 1919, 1919, 1919);
      DiagnosticType diagnosticType0 = ProcessClosurePrimitives.LATE_PROVIDE_ERROR;
      JSError jSError0 = JSError.make("JSC_LATE_PROVIDE_ERROR", node0, diagnosticType0, (String[]) null);
      String string0 = lightweightMessageFormatter0.formatWarning(jSError0);
      assertEquals("JSC_LATE_PROVIDE_ERROR:1919: WARNING - required \"{0}\" namespace not provided yet\n", string0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      LightweightMessageFormatter.LineNumberingFormatter lightweightMessageFormatter_LineNumberingFormatter0 = new LightweightMessageFormatter.LineNumberingFormatter();
      String string0 = lightweightMessageFormatter_LineNumberingFormatter0.formatRegion((Region) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      LightweightMessageFormatter.LineNumberingFormatter lightweightMessageFormatter_LineNumberingFormatter0 = new LightweightMessageFormatter.LineNumberingFormatter();
      SimpleRegion simpleRegion0 = new SimpleRegion(694, 694, "");
      String string0 = lightweightMessageFormatter_LineNumberingFormatter0.formatRegion(simpleRegion0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      LightweightMessageFormatter.LineNumberingFormatter lightweightMessageFormatter_LineNumberingFormatter0 = new LightweightMessageFormatter.LineNumberingFormatter();
      SimpleRegion simpleRegion0 = new SimpleRegion(19, 19, "[lS");
      String string0 = lightweightMessageFormatter_LineNumberingFormatter0.formatRegion(simpleRegion0);
      assertNotNull(string0);
      assertEquals("  19| [lS", string0);
  }
}
