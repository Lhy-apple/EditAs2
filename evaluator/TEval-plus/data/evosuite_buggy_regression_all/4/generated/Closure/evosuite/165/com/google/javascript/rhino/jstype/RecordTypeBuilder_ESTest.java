/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:26:47 GMT 2023
 */

package com.google.javascript.rhino.jstype;

import org.junit.Test;
import static org.junit.Assert.*;
import com.google.javascript.rhino.ErrorReporter;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.AllType;
import com.google.javascript.rhino.jstype.FunctionType;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeNative;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import com.google.javascript.rhino.jstype.RecordTypeBuilder;
import java.util.LinkedList;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class RecordTypeBuilder_ESTest extends RecordTypeBuilder_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null, false);
      RecordTypeBuilder recordTypeBuilder0 = new RecordTypeBuilder(jSTypeRegistry0);
      JSType jSType0 = recordTypeBuilder0.build();
      assertFalse(jSType0.isConstructor());
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      AllType allType0 = new AllType(jSTypeRegistry0);
      Node node0 = Node.newString("T");
      RecordTypeBuilder recordTypeBuilder0 = new RecordTypeBuilder(jSTypeRegistry0);
      recordTypeBuilder0.addProperty("Not declared as a constructor", allType0, node0);
      JSType jSType0 = recordTypeBuilder0.build();
      assertFalse(jSType0.isNoObjectType());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      JSTypeNative jSTypeNative0 = JSTypeNative.U2U_CONSTRUCTOR_TYPE;
      LinkedList<JSType> linkedList0 = new LinkedList<JSType>();
      Node node0 = jSTypeRegistry0.createParameters((List<JSType>) linkedList0);
      FunctionType functionType0 = jSTypeRegistry0.getNativeFunctionType(jSTypeNative0);
      RecordTypeBuilder recordTypeBuilder0 = new RecordTypeBuilder(jSTypeRegistry0);
      recordTypeBuilder0.addProperty(":<F?/(pk-H=X", functionType0, (Node) null);
      RecordTypeBuilder recordTypeBuilder1 = recordTypeBuilder0.addProperty(":<F?/(pk-H=X", functionType0, node0);
      assertNull(recordTypeBuilder1);
  }
}
