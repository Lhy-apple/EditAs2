/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:36:00 GMT 2023
 */

package com.fasterxml.jackson.databind.ser;

import org.junit.Test;
import static org.junit.Assert.*;
import com.fasterxml.jackson.databind.introspect.AnnotatedMember;
import com.fasterxml.jackson.databind.ser.AnyGetterWriter;
import com.fasterxml.jackson.databind.ser.BeanPropertyWriter;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.ser.std.MapSerializer;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class AnyGetterWriter_ESTest extends AnyGetterWriter_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      AnyGetterWriter anyGetterWriter0 = new AnyGetterWriter(beanPropertyWriter0, (AnnotatedMember) null, (MapSerializer) null);
      anyGetterWriter0.resolve(defaultSerializerProvider_Impl0);
  }
}