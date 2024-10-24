/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:24:43 GMT 2023
 */

package com.google.javascript.rhino.jstype;

import org.junit.Test;
import static org.junit.Assert.*;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import com.google.javascript.rhino.jstype.NamedType;
import com.google.javascript.rhino.jstype.RecordTypeBuilder;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class RecordTypeBuilder_ESTest extends RecordTypeBuilder_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      RecordTypeBuilder recordTypeBuilder0 = new RecordTypeBuilder(jSTypeRegistry0);
      Node node0 = Node.newString(1, "Not declared as a type name", (-820), 0);
      RecordTypeBuilder recordTypeBuilder1 = recordTypeBuilder0.addProperty("Named type with empty name component", (JSType) null, node0);
      JSType jSType0 = recordTypeBuilder1.build();
      assertFalse(jSType0.isCheckedUnknownType());
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      RecordTypeBuilder recordTypeBuilder0 = new RecordTypeBuilder(jSTypeRegistry0);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "com.google.common.base.Predicates$InstanceOfPredicate", ";H", (-1146), 3000);
      recordTypeBuilder0.addProperty("Unknown class name", namedType0, (Node) null);
      RecordTypeBuilder recordTypeBuilder1 = recordTypeBuilder0.addProperty("Unknown class name", (JSType) null, (Node) null);
      assertNull(recordTypeBuilder1);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      RecordTypeBuilder recordTypeBuilder0 = new RecordTypeBuilder(jSTypeRegistry0);
      JSType jSType0 = recordTypeBuilder0.build();
      assertTrue(jSType0.isNominalType());
  }
}
