/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 03:19:17 GMT 2023
 */

package com.fasterxml.jackson.databind.jsontype.impl;

import org.junit.Test;
import static org.junit.Assert.*;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.jsontype.impl.SubTypeValidator;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.time.chrono.ChronoLocalDate;
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
      SubTypeValidator subTypeValidator0 = SubTypeValidator.instance();
      JavaType javaType0 = TypeFactory.unknownType();
      subTypeValidator0.validateSubType((DeserializationContext) null, javaType0);
      assertEquals(0, javaType0.containedTypeCount());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      SubTypeValidator subTypeValidator0 = SubTypeValidator.instance();
      Class<ChronoLocalDate> class0 = ChronoLocalDate.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      subTypeValidator0.validateSubType((DeserializationContext) null, simpleType0);
      assertFalse(simpleType0.isMapLikeType());
  }
}