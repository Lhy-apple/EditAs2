/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:01:19 GMT 2023
 */

package com.google.javascript.rhino.jstype;

import org.junit.Test;
import static org.junit.Assert.*;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.FunctionType;
import com.google.javascript.rhino.jstype.IndexedType;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import com.google.javascript.rhino.jstype.RecordTypeBuilder;
import com.google.javascript.rhino.jstype.TemplateType;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class RecordTypeBuilder_ESTest extends RecordTypeBuilder_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      RecordTypeBuilder recordTypeBuilder0 = new RecordTypeBuilder(jSTypeRegistry0);
      JSType jSType0 = recordTypeBuilder0.build();
      assertTrue(jSType0.isObject());
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      RecordTypeBuilder recordTypeBuilder0 = new RecordTypeBuilder(jSTypeRegistry0);
      Node node0 = Node.newString(0, "com.google.javascript.jscomp.type.SemanticReverseAbstractInterpreter$RestrictByTrueInstanceOfResultVisitor", 0, 0);
      RecordTypeBuilder recordTypeBuilder1 = recordTypeBuilder0.addProperty("com.google.javascript.jscomp.type.SemanticReverseAbstractInterpreter$RestrictByTrueInstanceOfResultVisitor", (JSType) null, node0);
      JSType jSType0 = recordTypeBuilder1.build();
      assertFalse(jSType0.isEnumElementType());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      Node node0 = Node.newString((-99), "");
      TemplateType templateType0 = new TemplateType(jSTypeRegistry0, "Y`#oG$");
      IndexedType indexedType0 = new IndexedType(jSTypeRegistry0, templateType0, templateType0);
      FunctionType functionType0 = indexedType0.getOwnerFunction();
      RecordTypeBuilder recordTypeBuilder0 = new RecordTypeBuilder(jSTypeRegistry0);
      recordTypeBuilder0.addProperty((String) null, functionType0, node0);
      RecordTypeBuilder recordTypeBuilder1 = recordTypeBuilder0.addProperty((String) null, (JSType) null, node0);
      assertNull(recordTypeBuilder1);
  }
}