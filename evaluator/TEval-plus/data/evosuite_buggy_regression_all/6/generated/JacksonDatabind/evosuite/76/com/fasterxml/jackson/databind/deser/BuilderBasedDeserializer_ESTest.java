/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:27:53 GMT 2023
 */

package com.fasterxml.jackson.databind.deser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.databind.deser.BuilderBasedDeserializer;
import com.fasterxml.jackson.databind.util.NameTransformer;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class BuilderBasedDeserializer_ESTest extends BuilderBasedDeserializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      NameTransformer nameTransformer0 = NameTransformer.simpleTransformer((String) null, "");
      NameTransformer.Chained nameTransformer_Chained0 = new NameTransformer.Chained(nameTransformer0, nameTransformer0);
      BuilderBasedDeserializer builderBasedDeserializer0 = null;
      try {
        builderBasedDeserializer0 = new BuilderBasedDeserializer((BuilderBasedDeserializer) null, nameTransformer_Chained0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.BeanDeserializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      BuilderBasedDeserializer builderBasedDeserializer0 = null;
      try {
        builderBasedDeserializer0 = new BuilderBasedDeserializer((BuilderBasedDeserializer) null, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.BeanDeserializerBase", e);
      }
  }
}