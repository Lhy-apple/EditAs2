/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 03:19:16 GMT 2023
 */

package com.fasterxml.jackson.databind.jsontype.impl;

import org.junit.Test;
import static org.junit.Assert.*;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.jsontype.impl.SubTypeValidator;
import com.fasterxml.jackson.databind.type.TypeFactory;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class SubTypeValidator_ESTest extends SubTypeValidator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      SubTypeValidator subTypeValidator0 = new SubTypeValidator();
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      SubTypeValidator subTypeValidator0 = SubTypeValidator.instance();
      JavaType javaType0 = TypeFactory.unknownType();
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      subTypeValidator0.validateSubType(defaultDeserializationContext_Impl0, javaType0);
      assertFalse(javaType0.isCollectionLikeType());
  }
}