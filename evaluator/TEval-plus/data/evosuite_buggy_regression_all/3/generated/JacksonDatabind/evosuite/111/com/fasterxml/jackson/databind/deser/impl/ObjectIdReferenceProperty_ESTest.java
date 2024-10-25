/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:46:53 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.impl;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.Module;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.deser.NullValueProvider;
import com.fasterxml.jackson.databind.deser.SettableBeanProperty;
import com.fasterxml.jackson.databind.deser.UnresolvedForwardReference;
import com.fasterxml.jackson.databind.deser.impl.ObjectIdReferenceProperty;
import com.fasterxml.jackson.databind.introspect.ObjectIdInfo;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ObjectIdReferenceProperty_ESTest extends ObjectIdReferenceProperty_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse();
      UnresolvedForwardReference unresolvedForwardReference0 = new UnresolvedForwardReference(jsonParser0, (String) null);
      Class<Module> class0 = Module.class;
      ObjectIdReferenceProperty.PropertyReferring objectIdReferenceProperty_PropertyReferring0 = new ObjectIdReferenceProperty.PropertyReferring((ObjectIdReferenceProperty) null, unresolvedForwardReference0, class0, class0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      ObjectIdReferenceProperty objectIdReferenceProperty0 = null;
      try {
        objectIdReferenceProperty0 = new ObjectIdReferenceProperty((SettableBeanProperty) null, (ObjectIdInfo) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.introspect.ConcreteBeanPropertyBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.construct((String) null, (String) null);
      ObjectIdReferenceProperty objectIdReferenceProperty0 = null;
      try {
        objectIdReferenceProperty0 = new ObjectIdReferenceProperty((ObjectIdReferenceProperty) null, propertyName0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.introspect.ConcreteBeanPropertyBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      ObjectIdReferenceProperty objectIdReferenceProperty0 = null;
      try {
        objectIdReferenceProperty0 = new ObjectIdReferenceProperty((ObjectIdReferenceProperty) null, (JsonDeserializer<?>) null, (NullValueProvider) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.introspect.ConcreteBeanPropertyBase", e);
      }
  }
}
